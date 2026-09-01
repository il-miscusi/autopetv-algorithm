ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.base.name="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" \
      org.opencontainers.image.base.digest="sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"

# ─────────────────────────────────────────────────────────────────────────────
# AutoPET V 2026 —— Champion M0 + LocalEdit--TACE HoleGuard Fusion v5
#
# ⚠️ 关键设计：本容器必须 bake *autoPET-interactive fork 包*，而不是 PyPI 的
#    nnunetv2。因为推理依赖的这些类全是 fork 私有的：
#      - nnunetv2.inference.autopet_predictor.autoPETPredictor
#      - nnunetv2.training.dataloading.utils.sparse_to_dense_point_nnInteractive
#      - nnunetv2.training.dataloading.nnInteractive_clicks.PointInteraction_stub  (EDT 编码)
#      - nnunetv2.architecture.ResidualEncoderUNetOrgan.ResidualEncoderUNetOrgan
#    装 PyPI nnunetv2 会缺这些类，推理直接崩。
# ─────────────────────────────────────────────────────────────────────────────

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# ── 1. 业务依赖 (autoPET-interactive 之外的额外/锁版本依赖) ──────────────────
COPY --chown=algorithm:algorithm requirements.txt constraints.txt /opt/algorithm/
COPY --chown=algorithm:algorithm wheels /opt/algorithm/wheels
ARG OFFLINE_QA=0
RUN if [ "$OFFLINE_QA" = "1" ]; then \
      python -m pip install --user --no-index /opt/algorithm/wheels/*.whl; \
    else \
      python -m pip install --user -r requirements.txt -c constraints.txt; \
    fi

# ── 2. bake 已冻结的 autoPET-interactive fork 并 editable 安装 ──────────────
COPY --chown=algorithm:algorithm autoPET-interactive /opt/algorithm/autoPET-interactive
RUN if [ "$OFFLINE_QA" = "1" ]; then \
      python -m pip install --user --no-deps --no-build-isolation -e /opt/algorithm/autoPET-interactive; \
    else \
      python -m pip install --user -e /opt/algorithm/autoPET-interactive -c constraints.txt; \
    fi

# ── 3. 容器代码入口 ────────────────────────────────────────────────────────
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_gate.py /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_logreg.json /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_router.py /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_router_ensemble_v1.npz /opt/algorithm/
COPY --chown=algorithm:algorithm psma_shape_component_gate.py /opt/algorithm/
COPY --chown=algorithm:algorithm psma_shape_component_gate_final.json /opt/algorithm/
COPY --chown=algorithm:algorithm download_weights.py /opt/algorithm/
COPY --chown=algorithm:algorithm WEIGHTS_MANIFEST.json /opt/algorithm/
COPY --chown=algorithm:algorithm gaussian_probability_click.py /opt/algorithm/
COPY --chown=algorithm:algorithm pf_tace_runtime.py /opt/algorithm/
COPY --chown=algorithm:algorithm localedit_tace_gate.py /opt/algorithm/

# The AutoPET-III champion uses an incompatible nnunetv2 fork. Keep its source
# uninstalled and invoke it in an isolated subprocess with a dedicated PYTHONPATH.
COPY --chown=algorithm:algorithm champion /opt/app/champion

# ── 4. bake 冠军与冻结 LocalEdit FDG/PSMA 五折 checkpoint ───────────────
USER root
ARG CHAMPION_WEIGHTS_URL=https://zenodo.org/api/records/14007247/files/autoPET-3-LesionTracer.zip/content
ARG CHAMPION_WEIGHTS_MD5=566016409b0bd14770c0b57c1f2873f1
RUN set -eux; \
    mkdir -p /opt/app/champion/model; \
    python /opt/algorithm/download_weights.py "${CHAMPION_WEIGHTS_URL}" /tmp/champion.zip; \
    test "$(md5sum /tmp/champion.zip | cut -d' ' -f1)" = "${CHAMPION_WEIGHTS_MD5}"; \
    python -c "import zipfile; zipfile.ZipFile('/tmp/champion.zip').extractall('/opt/app/champion/model')"; \
    rm /tmp/champion.zip; \
    test "$(find /opt/app/champion/model -name checkpoint_final.pth | wc -l)" -eq 5; \
    chown -R algorithm:algorithm /opt/app/champion
USER algorithm

# nnUNet 环境变量 (推理用，raw/preprocessed 在容器内用不到但保留以防 import 时检查)
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

# GC 容器无网络 —— 权重必须已 bake；HF/任何联网调用都不可用
ENV nnUNet_compile="0"

# Some QA/base images already contain an unrelated nnunetv2 installation in
# /opt/conda. Put the frozen interactive fork first so the editable install can
# never be shadowed by that preinstalled package. The champion subprocess
# replaces PYTHONPATH with its own incompatible fork on purpose.
ENV PYTHONPATH="/opt/algorithm/autoPET-interactive"

ENTRYPOINT ["python", "-m", "process"]
