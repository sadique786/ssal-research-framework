from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for the SSAL research framework."""

    print("=" * 60)
    print("SSAL Research Framework")
    print("=" * 60)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
