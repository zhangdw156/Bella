from __future__ import annotations

DEFAULT_FC_MODEL_ALIAS = "gpt-4o-2024-11-20-FC"


def ensure_bfcl_model_alias(
    registry_name: str, base_model: str = DEFAULT_FC_MODEL_ALIAS
) -> None:
    """
    Ensure a BFCL model config exists for a custom registry_name.

    BFCL's evaluator requires all model names passed to eval_runner.main
    to exist in MODEL_CONFIG_MAPPING. To support a custom registry_name
    (e.g. 'bella-mvp') without 修改 BFCL 官方源码，我们在运行期为其创建
    一个轻量 alias，复用某个已有 FC 模型（默认 gpt-4o-2024-11-20-FC）
    的配置：

    - 不修改 BFCL 的 handler / checker / scoring 逻辑；
    - 只是让 evaluator 能识别该自定义模型名，并在 result/score 目录中
      使用与 registry_name 一致的子目录结构。
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    if registry_name in MODEL_CONFIG_MAPPING:
        return

    if base_model not in MODEL_CONFIG_MAPPING:
        raise RuntimeError(
            f"BFCL model config for base model '{base_model}' not found; "
            f"cannot create alias for custom registry name '{registry_name}'."
        )

    MODEL_CONFIG_MAPPING[registry_name] = MODEL_CONFIG_MAPPING[base_model]

