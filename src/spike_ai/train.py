"""Train các model theo một file config.

Cách dùng:
    python -m spike_ai.train --config configs/baseline_v0.yaml           # train, đánh giá trên dev / CV
    python -m spike_ai.train --config configs/human_v1.yaml --final      # đo thêm trên holdout

Chế độ đánh giá (split.dev_mode trong config), xem ADR-004:
    single  fit trên train, đo trên một tập dev (chia bằng hash). Hợp khi có nhiều trận.
    kfold   cross-validation theo nhóm trên toàn bộ dữ liệu không thuộc holdout.
            Báo cáo trung bình ± độ lệch chuẩn. Hợp khi ít trận.
    lopo    leave-one-player-out: mỗi fold bỏ ra một người chơi. Trả lời câu hỏi
            "model có tổng quát sang người chơi mới không?".

Nhãn có điều kiện (`conditional: true` trong schema, vd `spike_choice`):
    chỉ xác định ở một phần frame, các frame khác là NaN và bị loại khỏi train/đo/holdout.
    Dùng khi muốn trả lời câu hỏi hẹp "đã quyết định đập thì đập cú nào" thay vì
    "mỗi frame nên làm gì" - tránh mất cân bằng 0,16% so với 99,84% frame None.

Holdout (chỉ đo khi có --final, và chỉ khi đã CHỐT model):
    split.holdout_ratio > 0   holdout = các nhóm được hash chọn ra từ data_dirs
    holdout_dirs: [...]       holdout "tương lai" = dữ liệu thu SAU khi chốt model, để ở thư mục riêng

Kết quả:
    models/<run_name>_<thời gian>/
        <target>__<model>.joblib   # model + metadata (feature, class, golden vectors)
        metrics.json               # toàn bộ chỉ số
        split.json                 # nhóm nào ở tập nào + báo cáo cân bằng + cảnh báo
        config.yaml                # bản sao config để tái lập
        cm_<tập đo>_<target>__<model>.png
    reports/experiments.md         # thêm 1 dòng cho mỗi (tập đo, target, model)
"""

import argparse
import inspect
import json
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from . import models as model_zoo
from .data import load_dirs
from .evaluate import compute_metrics, plot_confusion_matrix, summarize_folds
from .features import build
from .paths import MODELS_DIR, REPORTS_DIR, resolve
from .schema import feature_names, load_spec
from .split import (
    DEV,
    HOLDOUT,
    TRAIN,
    assert_no_overlap,
    cv_folds,
    group_keys,
    split_by_hash,
    split_manifest,
    split_report,
)

N_GOLDEN = 20  # số vector mẫu dùng để kiểm tra C++ ra cùng kết quả với Python
DEV_MODES = ("single", "kfold", "lopo", "none")


def load_config(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(dirs, version: str, game_versions: list[str] | None) -> pd.DataFrame:
    """Đọc log; nếu config có game_versions thì chỉ giữ log từ các phiên bản game đó (schema v1)."""
    df = load_dirs(dirs, expected_version=version)
    if game_versions:
        if "game_version" not in df.columns:
            raise ValueError("game_versions chỉ dùng được với log có cột game_version (schema v1)")
        df = df[df["game_version"].astype(str).isin([str(v) for v in game_versions])].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Không còn dòng nào sau khi lọc game_versions={game_versions}")
    return df


def validate_split_config(cfg: dict) -> dict:
    s = dict(cfg["split"])
    s.setdefault("group_by", "match")
    s.setdefault("dev_mode", "single")
    s.setdefault("dev_ratio", 0.0)
    s.setdefault("holdout_ratio", 0.0)
    s.setdefault("n_folds", 5)
    if s["dev_mode"] not in DEV_MODES:
        raise ValueError(f"split.dev_mode phải là một trong {DEV_MODES}")
    if s["dev_mode"] == "single" and s["dev_ratio"] <= 0:
        raise ValueError("dev_mode=single cần dev_ratio > 0")
    if s["dev_mode"] != "single" and s["dev_ratio"] > 0:
        raise ValueError(f"dev_mode={s['dev_mode']} dùng cross-validation, hãy đặt dev_ratio: 0")
    if cfg.get("holdout_dirs") and s["holdout_ratio"] > 0:
        raise ValueError("Chỉ chọn một loại holdout: holdout_ratio > 0 HOẶC holdout_dirs")
    return s


def check_label_shift(cfg: dict, version: str) -> int:
    """Config log v1/v2 BẮT BUỘC khai báo `label_shift` (ADR-011).

    Game ghi log SAU khi đã áp hành động, nên ghép nhãn cùng frame (label_shift: 0) cho model nhìn
    thấy đáp án ngay trong feature: điểm offline cao ảo, bot trong game kém. Trước đây giá trị mặc
    định là 0, nên chạy lại một config cũ là ra model rò rỉ mà không ai biết. Giờ phải ghi rõ:
        label_shift: 1   cách đúng
        label_shift: 0   chỉ để tái lập kết quả cũ - in cảnh báo
    Log v0 ghi theo cách khác (log debug của MainScene) nên không áp dụng.
    """
    if version == "v0":
        return int(cfg.get("label_shift", 0))
    if "label_shift" not in cfg:
        raise ValueError(
            f"Config schema {version} phải khai báo label_shift (ADR-011). "
            "Dùng 'label_shift: 1'; 'label_shift: 0' chỉ để tái lập kết quả cũ (model rò rỉ nhãn)."
        )
    shift = int(cfg["label_shift"])
    if shift == 0:
        print("  ! label_shift: 0 -> model RÒ RỈ NHÃN (ADR-011): điểm offline cao ảo. Chỉ dùng để tái lập kết quả cũ.")
    return shift


def action_window_mask(df: pd.DataFrame, y: pd.DataFrame, half_width: int) -> np.ndarray:
    """Mặt nạ giữ các frame trong cửa sổ +/- half_width quanh mỗi frame CÓ hành động.

    Vì sao cần: trong log v1 thật, chỉ khoảng 0,7% frame có hành động - phần còn lại là
    `None` (đứng chờ bóng). Train trên toàn bộ frame vừa chậm vừa làm lớp `None` áp đảo.
    Lấy cửa sổ quanh lúc chạm bóng giữ LẠI 100% mẫu hành động mà cắt phần lớn frame `None`,
    đúng kỹ thuật ghi trong PLAN.md GĐ4.

    Cửa sổ KHÔNG tràn qua ranh giới trận: hai trận nối nhau trong cùng DataFrame, frame cuối
    trận này không liên quan gì tới frame đầu trận sau.
    """
    is_action = (y["action"].notna() & (y["action"] != "None")).to_numpy()
    keep = np.zeros(len(df), dtype=bool)

    for positions in df.groupby("match_id", sort=False).indices.values():
        local = is_action[positions]
        if not local.any():
            continue
        hits = np.flatnonzero(local)
        for h in hits:
            lo = max(0, h - half_width)
            hi = min(len(positions), h + half_width + 1)
            keep[positions[lo:hi]] = True
    return keep


CONTROLLER_PREFIX = {"player": "p_", "opponent": "o_"}
AGENT_TEAM = {"player": 0, "opponent": 1}  # Team::LEFT / RIGHT
BALL_ALIVE = -1
# Sau khi bong chet, diem duoc cong trong vong ngan nay frame (dem nguoc 50 -> 0 roi reset).
SCORE_LOOKAHEAD = 200


SPIKE_KINDS = (2, 7, 8, 9)  # ActionState: Spike, SpikeLight, SpikeMedium, SpikeStrong
AGENT_ENTITY = {"player": 0, "opponent": 3}


def kill_possession_mask(df: pd.DataFrame, agent: str) -> np.ndarray:
    """Frame thuộc LƯỢT CẦM BÓNG CUỐI của `agent` trong những pha nó TỰ GHI ĐIỂM bằng cú đập.

    "Tự ghi điểm" = đội của agent thắng pha bóng VÀ cú chạm cuối cùng của pha là cú đập của chính
    agent - không tính điểm thắng vì đối thủ đánh hỏng. "Lượt cầm bóng cuối" = từ ngay sau cú chạm
    cuối của đối thủ (hoặc đầu pha bóng nếu đối thủ chưa chạm) tới frame cú đập đó. Đây là chuỗi
    hành động thật sự tạo ra điểm (vd nhận bóng -> Set -> Jump -> Spike), phần còn lại của pha bóng
    không gắn trực tiếp với điểm nên không học (ADR-014).
    """
    from .data import ENTITY_NONE, rebuild_kind_v1_to_v2

    me = AGENT_ENTITY[agent]
    won = won_rally_mask(df, AGENT_TEAM[agent])
    out = np.zeros(len(df), dtype=bool)
    for positions in df.groupby("match_id", sort=False).indices.values():
        g = df.iloc[positions]
        lt = g["rally_last_touch"].fillna(ENTITY_NONE).astype(int).to_numpy()
        tc = g["rally_touch_count"].fillna(0).astype(int).to_numpy()
        kind = (
            g["rally_last_touch_kind"].fillna(0).astype(int).to_numpy()
            if "rally_last_touch_kind" in g
            else rebuild_kind_v1_to_v2(g).to_numpy()
        )
        touch = np.r_[False, (lt[1:] != lt[:-1]) | (tc[1:] != tc[:-1])] & (lt != ENTITY_NONE)
        alive = (g["ball_state_frame"].to_numpy() == BALL_ALIVE).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, alive, 0]))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            if not won[positions[start]]:
                continue
            touches = start + np.flatnonzero(touch[start:end])
            if not len(touches):
                continue
            last = touches[-1]
            if lt[last] != me or kind[last] not in SPIKE_KINDS:
                continue  # điểm thắng không đến từ cú đập của chính agent
            theirs = touches[lt[touches] != me]
            begin = theirs[-1] + 1 if len(theirs) else start
            out[positions[begin : last + 1]] = True
    return out


def won_rally_mask(df: pd.DataFrame, team: int) -> np.ndarray:
    """Frame nao thuoc mot pha bong (doan bong dang song) ma doi `team` THANG diem.

    Dung cho tu hoc (self_imitation): chi bat chuoc nhung pha bong bot ML ghi duoc diem.
    Nguoi thang = ben duoc cong diem ngay sau khi bong chet. Tinh rieng tung tran.
    """
    out = np.zeros(len(df), dtype=bool)
    mine_col, other_col = ("score_left", "score_right") if team == 0 else ("score_right", "score_left")
    for positions in df.groupby("match_id", sort=False).indices.values():
        alive = (df["ball_state_frame"].to_numpy()[positions] == BALL_ALIVE).astype(int)
        mine = df[mine_col].to_numpy()[positions]
        other = df[other_col].to_numpy()[positions]
        edges = np.flatnonzero(np.diff(np.r_[0, alive, 0]))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            after = slice(end - 1, min(end - 1 + SCORE_LOOKAHEAD, len(positions)))
            gained = np.flatnonzero((mine[after] > mine[end - 1]) | (other[after] > other[end - 1]))
            if len(gained) and mine[after][gained[0]] > mine[end - 1]:
                out[positions[start:end]] = True
    return out


def as_agent_list(agent) -> list[str]:
    """`agent` trong config co the la mot chuoi ("player") hoac mot danh sach."""
    return [agent] if isinstance(agent, str) else list(agent)


def build_rows(df: pd.DataFrame, version: str, agents, cfg: dict):
    """Dung (df, X, y) de train, gop tu MOT HAY NHIEU goc nhin (agent).

    Hai bo loc ap dung RIENG cho tung agent, roi moi gop lai:

    1. CHI GIU FRAME DO NGUOI DIEU KHIEN (`only_human`, mac dinh bat).
       Logs/v1 chua ca tran nguoi-vs-bot lan tran DAgger, trong do san trai la BOT LUAT.
       Khong loc thi train `agent: player` tren thu muc do se hoc tu bot luat.
       Dua vao cot <p_|o_>controller cua chinh agent (enum Controller: human = 0), nen
       chinh xac tung frame - o tran DAgger, san phai chi la nguoi o dung nhung doan
       nguoi choi gianh quyen. Log khong co cot controller (v0) thi bo qua buoc nay.

    2. CUA SO QUANH LUC CHAM BONG (`frames_around_action`).
       Tinh tren df DAY DU roi moi AND voi bo loc 1, vi cua so can cac frame lien tiep.

    Cot `_agent` duoc them de group_keys biet dong nao thuoc goc nhin nao. Hai goc nhin
    cua CUNG mot tran co cung match_id, nen luon roi vao cung mot fold - khong ro ri.
    """
    agents = as_agent_list(agents)
    only_human = cfg.get("only_human", True)
    weights = cfg.get("agent_weight") or {}
    shift = int(cfg.get("label_shift", 0))
    # Dong nao con dong ke tiep TRONG CUNG TRAN (de lay nhan cua dong do).
    by_match = df["match_id"].to_numpy()
    has_next = (df.groupby(by_match, sort=False).cumcount(ascending=False) >= shift).to_numpy() if shift else None
    window = cfg.get("frames_around_action")
    controllers = load_spec(version).get("enums", {}).get("Controller", {})
    human_code, ml_code = controllers.get("human", 0), controllers.get("ml_bot", 2)
    # Ai được coi là "người dạy" (nhãn đáng học). Mặc định chỉ người chơi; vòng tự học hướng B thêm
    # "expert" = chuyên gia ghi điểm điều khiển sân bot ML trong MatchRunner (ADR-015).
    teacher_names = cfg.get("train_on_controllers") or ["human"]
    unknown = [n for n in teacher_names if n not in controllers and n != "human"]
    if unknown:
        raise ValueError(f"train_on_controllers có giá trị lạ {unknown}; có: {sorted(controllers)}")
    teacher_codes = [controllers.get(n, human_code) for n in teacher_names]
    controller_weight = {controllers[n]: float(w) for n, w in (cfg.get("controller_weight") or {}).items()}
    si = cfg.get("self_imitation")
    if si and si.get("agent") not in agents:
        raise ValueError(f"self_imitation.agent = {si.get('agent')} phai nam trong agent = {agents}")

    parts_df, parts_X, parts_y = [], [], []
    for agent in agents:
        X, y = build(df, version, agent)
        keep = np.ones(len(df), dtype=bool)
        notes = []
        col = f"{CONTROLLER_PREFIX[agent]}controller"
        controller = df[col] if col in df.columns else None

        if shift:
            # GHEP NHAN VAO DUNG TRANG THAI LUC RA QUYET DINH (label_shift, xem ADR-011).
            #
            # Game ghi log SAU khi da ap hanh dong cua frame do: trong GenerateIntent::update,
            # cooldown / bat dau nhay / danh bong deu chay TRUOC khi MainScene ghi log. Do tren
            # 2.871 lan bam phim (120 tran): o dong co nhan, nhan vat da vao cooldown 100%, bong
            # da mang quy dao cu danh 100%, nguoi cham cuoi da la "minh" 100% - model nhin thay
            # dap an ngay trong feature. Dong ngay truoc (k-1) moi la trang thai luc quyet dinh:
            # nhan vat ranh tay 99,2-100%, bong chua bi danh.
            #
            # Nen: feature cua dong k ghep voi nhan cua dong k+shift, trong cung tran. Ai ra
            # quyet dinh cung lay o dong k+shift (vd lan gianh quyen DAgger bat dau dung frame do).
            y = y.groupby(by_match, sort=False).shift(-shift)
            if controller is not None:
                controller = controller.groupby(by_match, sort=False).shift(-shift)
            keep &= has_next
            notes.append(f"ghep nhan lui {shift} frame")

        self_rows = np.zeros(len(df), dtype=bool)
        if only_human and controller is not None:
            human = controller.isin(teacher_codes).to_numpy()
            notes.append(f"{'+'.join(teacher_names)} dieu khien {human.sum():,}")
            if cfg.get("expert_only_kills") and "expert" in controllers:
                # Chỉ học những lượt cầm bóng CHUYÊN GIA TỰ GHI ĐIỂM bằng cú đập (kill_possession_mask):
                # chuyên gia tạo nhiều kiểu ghi điểm (đập nhanh, Set -> Jump -> Spike), bot ML chỉ chép
                # những lượt thành công, không chép lượt đập bị đỡ. Frame người chơi không bị lọc.
                expert = (controller == controllers["expert"]).to_numpy()
                human = human & (~expert | kill_possession_mask(df, agent))
                notes.append(f"chuyen gia chi luot ghi diem -> con {human.sum():,}")
            allowed = human
            if si and agent == si["agent"]:
                # TU HOC (self-imitation): them frame cua CHINH bot ML trong cac tran tu sinh
                # (player_id = si.player_id, vd MLAUTO do MatchRunner ghi), chi o nhung pha bong
                # phia no thang diem. Log nguoi vs bot cu co player_id khac nen khong bi lan vao.
                pid = df[f"{CONTROLLER_PREFIX[agent]}player_id"].astype(str).to_numpy() == str(si["player_id"])
                self_rows = pid & (controller == ml_code).to_numpy()
                mode = si.get("mode", "won_rallies" if si.get("only_won_rallies", True) else "all")
                if mode == "kills":
                    self_rows &= kill_possession_mask(df, agent)
                elif mode == "won_rallies":
                    self_rows &= won_rally_mask(df, AGENT_TEAM[agent])
                elif mode != "all":
                    raise ValueError(f"self_imitation.mode = {mode}; chỉ có kills / won_rallies / all")
                notes.append(f"tu hoc {self_rows.sum():,}")
                allowed = human | self_rows
            keep &= allowed

        if window:
            in_window = action_window_mask(df, y, int(window))
            notes.append(f"cua so +/-{window} {in_window.sum():,}")
            keep &= in_window

        print(f"  agent={agent}: giu {keep.sum():,}/{len(df):,} frame ({', '.join(notes) or 'khong loc'})")
        weight = np.full(len(df), float(weights.get(agent, 1.0)))
        if controller is not None:
            # Trọng số theo người dạy (vd chuyên gia), ghi đè agent_weight cho đúng các dòng đó.
            for code, w in controller_weight.items():
                weight[(controller == code).to_numpy()] = w
            if cfg.get("expert_kill_weight") is not None and "expert" in controllers:
                # Học CHỦ YẾU từ pha ghi điểm: lượt chuyên gia tự ghi điểm bằng cú đập (kill_possession_mask)
                # có trọng số riêng, cao hơn các lượt chuyên gia còn lại (vẫn học, theo controller_weight).
                kills = (controller == controllers["expert"]).to_numpy() & kill_possession_mask(df, agent)
                weight[kills] = float(cfg["expert_kill_weight"])
                notes.append(f"luot chuyen gia ghi diem x{cfg['expert_kill_weight']} {kills.sum():,}")
        if si and agent == si["agent"]:
            # Trong so RIENG cho dong tu hoc, khong nhan voi agent_weight (agent_weight cua san phai
            # danh cho frame nguoi gianh quyen DAgger, vd x20).
            weight[self_rows] = float(si.get("weight", 1.0))
        parts_df.append(df[keep].assign(_agent=agent, _weight=weight[keep]))
        parts_X.append(X[keep])
        part_y = y[keep].copy()
        # groupby().shift() doi move sang so thuc (1.0) vi co NaN o cuoi tran; cac dong giu lai
        # khong con NaN nen ep ve so nguyen, de ten lop van la -1/0/1 nhu spec.
        if shift and "move" in part_y and part_y["move"].notna().all():
            part_y["move"] = part_y["move"].astype(int)
        parts_y.append(part_y)

    out_df = pd.concat(parts_df, ignore_index=True)
    out_X = pd.concat(parts_X, ignore_index=True)
    out_y = pd.concat(parts_y, ignore_index=True)
    if out_df.empty:
        raise ValueError(
            f"Khong con frame nao de train sau khi loc (agents={agents}, only_human={only_human}). "
            "Kiem cot controller trong log, hoac dat only_human: false neu co y train tren bot."
        )
    return out_df, out_X, out_y


def uses_weights(cfg: dict) -> bool:
    """Config co dat trong so tung dong khong (agent_weight hoac self_imitation.weight)."""
    return bool(
        cfg.get("agent_weight") or cfg.get("controller_weight") or (cfg.get("self_imitation") or {}).get("weight")
    )


def _fit(name, seed, params, X, y, w):
    """Tao model va fit. `w` = trong so tung dong (None = khong dung trong so).

    Dung cho DAgger: du lieu gianh quyen chi chiem vai phan tram so dong, nen bi du lieu
    cu nhan chim - do duoc vong 1 khong doi duoc hanh vi bot (argmax ra Set/Jump/Bump o 0
    frame truoc va sau). Xem config agent_weight.
    """
    model = model_zoo.create(name, seed, params)
    if w is None:
        return model.fit(X, y)
    final_step = model.steps[-1] if hasattr(model, "steps") else None
    est = final_step[1] if final_step else model
    if "sample_weight" not in inspect.signature(est.fit).parameters:
        raise ValueError(f"model '{name}' khong ho tro sample_weight - bo no khoi config co agent_weight")
    if final_step:
        return model.fit(X, y, **{f"{final_step[0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def run(config_path: str, final: bool = False) -> dict:
    cfg = load_config(config_path)
    version, agent, seed = cfg["schema"], cfg["agent"], cfg.get("seed", 42)
    s = validate_split_config(cfg)
    spec = load_spec(version)
    check_label_shift(cfg, version)

    df = load_data(cfg["data_dirs"], version, cfg.get("game_versions"))
    # Loc nguoi dieu khien + cua so quanh luc cham bong, cho tung agent roi gop lai.
    # Phai lam TRUOC khi chia tap, neu khong chi so dong cua split khong con khop voi X/y.
    excluded = set(cfg.get("exclude_matches") or [])
    if excluded:
        drop = df["match_id"].isin(excluded)
        n_drop = df.loc[drop, "match_id"].nunique()
        print(f"  exclude_matches: bo {n_drop}/{len(excluded)} tran ({int(drop.sum()):,} frame)")
        df = df[~drop].reset_index(drop=True)
    df, X, y = build_rows(df, version, agent, cfg)
    W = df["_weight"].to_numpy() if uses_weights(cfg) else None
    if W is not None:
        for a in as_agent_list(agent):
            m = (df["_agent"] == a).to_numpy()
            print(f"  agent_weight {a}: x{W[m][0] if m.any() else 0:g} tren {m.sum():,} dong -> tong {W[m].sum():,.0f}")

    groups = group_keys(df, s["group_by"], agent)
    index = split_by_hash(groups, s["salt"], s["dev_ratio"], s["holdout_ratio"])

    # ---- holdout: chỉ nạp / đụng tới khi --final
    holdout = None
    if final:
        holdout = _load_holdout(cfg, s, df, X, y, index, version, agent)

    # ---- báo cáo cân bằng (không mô tả holdout nếu chưa --final)
    visible = {k: v for k, v in index.items() if k != HOLDOUT}
    report, warnings = split_report(groups, visible, {t: y[t] for t in cfg["targets"]})

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / f"{cfg['run_name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(config_path), out_dir / "config.yaml")

    has_holdout = bool(len(index[HOLDOUT]) or cfg.get("holdout_dirs"))
    _print_split(version, agent, s, report, warnings, holdout, has_holdout)
    all_metrics, log_rows = {}, []

    for target in cfg["targets"]:
        order = spec["labels"][target]["values"]
        rows = np.flatnonzero(y[target].notna().to_numpy())  # nhãn có điều kiện -> bỏ dòng không áp dụng
        keep = {n: np.intersect1d(idx, rows) for n, idx in index.items()}
        _print_target(target, spec, y[target], rows, len(X))
        _require_samples(target, s, keep)

        for name, params in cfg["models"].items():
            key = f"{target}__{name}"

            if s["dev_mode"] == "none":
                # Chỉ fit trên TOÀN BỘ dữ liệu không thuộc holdout, không đo offline. Dùng khi việc chấm
                # điểm làm ở chỗ khác (vòng tự học đo trong game) và model phải được train ngang hàng
                # với model đem so (vd model xuất phát fit trên 100% dữ liệu).
                fit_idx = np.union1d(keep[TRAIN], keep[DEV])
                model = _fit(
                    name, seed, params, X.iloc[fit_idx], y[target].iloc[fit_idx], None if W is None else W[fit_idx]
                )
                eval_name, golden, m = "none", X.iloc[fit_idx].head(N_GOLDEN), None
                print(f"  [fit    ] {key:<32} fit trên {len(fit_idx):,} dòng, không đo offline (dev_mode: none)")
            elif s["dev_mode"] == "single":
                fit_idx = keep[TRAIN]
                model = _fit(
                    name, seed, params, X.iloc[fit_idx], y[target].iloc[fit_idx], None if W is None else W[fit_idx]
                )
                dev_idx = keep[DEV]
                m = compute_metrics(y[target].iloc[dev_idx], model.predict(X.iloc[dev_idx]), order)
                eval_name, golden = "dev", X.iloc[dev_idx].head(N_GOLDEN)
                n_train, n_eval = len(fit_idx), len(dev_idx)
            else:
                fit_idx = np.union1d(keep[TRAIN], keep[DEV])
                m = _cross_validate(X, y[target], df, agent, s, fit_idx, name, params, seed, order, W)
                # model cuối cùng (để export / đo holdout) fit trên toàn bộ dữ liệu không thuộc holdout
                model = _fit(
                    name, seed, params, X.iloc[fit_idx], y[target].iloc[fit_idx], None if W is None else W[fit_idx]
                )
                eval_name = "cv" if s["dev_mode"] == "kfold" else "lopo"
                golden = X.iloc[fit_idx].head(N_GOLDEN)
                n_train, n_eval = len(fit_idx), len(fit_idx)

            if m is not None:
                all_metrics[f"{eval_name}/{key}"] = m
                plot_confusion_matrix(m, f"{eval_name}: {key}", out_dir / f"cm_{eval_name}_{key}.png")
                log_rows.append((eval_name, target, name, n_train, n_eval, m))
                _print_result(eval_name, key, m)

            if holdout is not None:
                X_h, y_h = holdout
                h = y_h[target].notna().to_numpy()
                mh = compute_metrics(y_h[target][h], model.predict(X_h[h]), order)
                all_metrics[f"holdout/{key}"] = mh
                plot_confusion_matrix(mh, f"holdout: {key}", out_dir / f"cm_holdout_{key}.png")
                log_rows.append(("holdout", target, name, n_train, int(h.sum()), mh))
                _print_result("holdout", key, mh)

            joblib.dump(
                {
                    "model": model,
                    "model_name": name,
                    "target": target,
                    "agent": "+".join(as_agent_list(agent)),
                    "schema": version,
                    "features": feature_names(spec),
                    "classes": [str(c) for c in model.classes_],
                    "golden_X": golden.to_numpy().tolist(),
                    "golden_proba": model.predict_proba(golden).tolist(),
                },
                out_dir / f"{key}.joblib",
            )

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        manifest = split_manifest(groups, index if final else visible)
        json.dump({"groups": manifest, "report": report, "warnings": warnings}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    _append_experiment_log(out_dir.name, version, agent, log_rows)
    print(f"Đã lưu vào {out_dir}")
    return all_metrics


def _cross_validate(X, y_target, df, agent, s, pool_idx, name, params, seed, order, W=None) -> dict:
    """CV trên pool_idx. Trả về chỉ số tính trên dự đoán out-of-fold, kèm trung bình ± std từng fold."""
    by = "player" if s["dev_mode"] == "lopo" else s["group_by"]
    cv_groups = group_keys(df, by, agent).iloc[pool_idx].reset_index(drop=True)
    X_pool, y_pool = X.iloc[pool_idx].reset_index(drop=True), y_target.iloc[pool_idx].reset_index(drop=True)
    W_pool = None if W is None else W[pool_idx]

    # mỗi dòng của pool được validate đúng một lần -> mọi giá trị đều bị ghi đè bằng dự đoán out-of-fold
    oof = y_pool.copy()
    fold_metrics = []
    for tr, va in cv_folds(cv_groups, s["dev_mode"], s["n_folds"]):
        model = _fit(name, seed, params, X_pool.iloc[tr], y_pool.iloc[tr], None if W_pool is None else W_pool[tr])
        pred = model.predict(X_pool.iloc[va])
        oof.iloc[va] = pred
        fold_metrics.append(compute_metrics(y_pool.iloc[va], pred, order))

    m = compute_metrics(y_pool, oof, order)  # confusion matrix gộp của mọi fold
    summary = summarize_folds(fold_metrics)
    m.update({k: summary[k] for k in ("accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "n_folds")})
    m["folds"] = [{"accuracy": f["accuracy"], "macro_f1": f["macro_f1"]} for f in fold_metrics]
    return m


def _load_holdout(cfg, s, df, X, y, index, version, agent):
    if cfg.get("holdout_dirs"):
        df_h = load_data(cfg["holdout_dirs"], version, cfg.get("game_versions"))
        assert_no_overlap(df["match_id"], df_h["match_id"])
        _, X_h, y_h = build_rows(df_h, version, agent, {**cfg, "frames_around_action": None})
        return X_h, y_h
    if s["holdout_ratio"] > 0:
        idx = index[HOLDOUT]
        return X.iloc[idx], y.iloc[idx]
    raise ValueError("Config này không có holdout (holdout_ratio = 0 và không có holdout_dirs) -> bỏ --final")


def _print_split(version, agent, s, report, warnings, holdout, has_holdout: bool):
    names = {TRAIN: "train" if s["dev_mode"] == "single" else "dữ liệu cross-validation"}
    parts = [f"{names.get(n, n)} {r['groups']} nhóm, {r['frames']} frame" for n, r in report.items()]
    print(f"Schema {version} | agent {agent} | group_by {s['group_by']} | dev_mode {s['dev_mode']}")
    print("  " + " / ".join(parts))
    if holdout is not None:
        print(f"  holdout: {len(holdout[0])} frame (đang đo vì --final)")
    elif has_holdout:
        print("  holdout được giữ kín - chỉ đo khi chạy với --final")
    else:
        print("  config này không có holdout")
    for w in warnings:
        print(f"  ! {w}")


def _require_samples(target, s, keep) -> None:
    """Báo lỗi dễ hiểu khi nhãn có điều kiện không còn mẫu, thay vì để sklearn ném traceback."""
    need = [TRAIN] if s["dev_mode"] != "single" else [TRAIN, DEV]
    empty = [n for n in need if len(keep[n]) == 0]
    if empty:
        raise ValueError(
            f"Nhãn '{target}' không có mẫu nào trong tập {empty}. "
            f"Chạy 'python scripts/count_spikes.py' để xem cần thu thêm bao nhiêu dữ liệu."
        )


def _print_target(target, spec, y_target, rows, n_all: int):
    """Với nhãn có điều kiện (vd spike_choice), in rõ còn lại bao nhiêu frame và mỗi lớp bao nhiêu mẫu."""
    if not spec["labels"][target].get("conditional"):
        return
    counts = y_target.iloc[rows].value_counts()
    per_class = ", ".join(f"{k} {v}" for k, v in counts.items())
    print(f"  {target}: nhãn có điều kiện -> {len(rows)}/{n_all} frame ({per_class or 'KHÔNG CÓ MẪU NÀO'})")


def _print_result(eval_name, key, m):
    std = f" ± {m['macro_f1_std']:.3f}" if "macro_f1_std" in m else ""
    print(f"  [{eval_name:<7}] {key:<32} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}{std}")
    if m["rare_labels"]:
        print(f"            ! lớp ít mẫu (< 10), F1 rất nhiễu: {m['rare_labels']}")


def _fmt(m: dict, key: str) -> str:
    std = m.get(f"{key}_std")
    return f"{m[key]:.3f} ± {std:.3f}" if std is not None else f"{m[key]:.3f}"


def _append_experiment_log(run_id, version, agent, rows) -> None:
    log = REPORTS_DIR / "experiments.md"
    today = datetime.now().strftime("%Y-%m-%d")
    with open(log, "a", encoding="utf-8") as f:
        for eval_name, target, name, n_train, n_eval, m in rows:
            f.write(
                f"| {today} | {run_id} | {version} | {agent} | {target} | {name} "
                f"| {eval_name} | {n_train} | {n_eval} | {_fmt(m, 'accuracy')} | {_fmt(m, 'macro_f1')} | |\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="đường dẫn file YAML trong configs/")
    parser.add_argument("--final", action="store_true", help="đo thêm trên holdout (chỉ khi đã chốt model)")
    args = parser.parse_args()
    run(args.config, final=args.final)


if __name__ == "__main__":
    main()
