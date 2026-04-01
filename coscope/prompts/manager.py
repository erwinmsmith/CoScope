"""
Prompt Manager for loading and managing prompt templates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .templates import PromptTemplate, RoleTemplates

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates with loading from files or inline definitions.

    Example usage:
    ```python
    manager = PromptManager()

    # Load from file
    manager.load_from_file("prompts.json")

    # Get a template
    template = manager.get_template("planner_retrieval")

    # Render with variables
    rendered = manager.render("planner_retrieval", query="What are constraints?")
    ```
    """

    def __init__(
        self,
        templates: Optional[Dict[str, PromptTemplate]] = None,
    ):
        """
        Initialize prompt manager.

        Args:
            templates: Optional initial templates
        """
        self._templates: Dict[str, PromptTemplate] = templates or {}

        # Load default role templates
        for name, tpl in RoleTemplates.all_templates().items():
            self._templates[f"role/{name}"] = tpl

    def add_template(self, template: PromptTemplate) -> None:
        """Add a template."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def render(
        self,
        name: str,
        default: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Render a template by name.

        Args:
            name: Template name
            default: Default string if template not found
            **kwargs: Variables for template

        Returns:
            Rendered string
        """
        template = self.get_template(name)
        if template:
            return template.render(**kwargs)

        if default:
            return default.format(**kwargs)

        logger.warning(f"Template not found: {name}")
        return name

    def load_from_file(
        self,
        path: Union[str, Path],
        format: str = "json",
    ) -> int:
        """
        Load templates from a file.

        Args:
            path: Path to template file
            format: File format (json, yaml)

        Returns:
            Number of templates loaded
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"Template file not found: {path}")
            return 0

        if format == "json":
            return self._load_json(path)
        elif format == "yaml":
            return self._load_yaml(path)
        else:
            logger.error(f"Unsupported format: {format}")
            return 0

    def _load_json(self, path: Path) -> int:
        """Load templates from JSON file."""
        with open(path) as f:
            data = json.load(f)

        count = 0
        if "templates" in data:
            for tpl_data in data["templates"]:
                tpl = PromptTemplate.from_dict(tpl_data)
                self.add_template(tpl)
                count += 1
        elif isinstance(data, dict):
            for name, tpl_data in data.items():
                if isinstance(tpl_data, dict):
                    tpl_data["name"] = name
                    tpl = PromptTemplate.from_dict(tpl_data)
                    self.add_template(tpl)
                    count += 1

        logger.info(f"Loaded {count} templates from {path}")
        return count

    def _load_yaml(self, path: Path) -> int:
        """Load templates from YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed")
            return 0

        with open(path) as f:
            data = yaml.safe_load(f)

        count = 0
        if "templates" in data:
            for tpl_data in data["templates"]:
                tpl = PromptTemplate.from_dict(tpl_data)
                self.add_template(tpl)
                count += 1

        logger.info(f"Loaded {count} templates from {path}")
        return count

    def save_to_file(
        self,
        path: Union[str, Path],
        format: str = "json",
    ) -> bool:
        """
        Save templates to a file.

        Args:
            path: Output path
            format: Output format (json, yaml)

        Returns:
            True if successful
        """
        path = Path(path)

        if format == "json":
            return self._save_json(path)
        elif format == "yaml":
            return self._save_yaml(path)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

    def _save_json(self, path: Path) -> bool:
        """Save templates to JSON file."""
        data = {
            "templates": [tpl.to_dict() for tpl in self._templates.values()]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._templates)} templates to {path}")
        return True

    def _save_yaml(self, path: Path) -> bool:
        """Save templates to YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed")
            return False

        data = {
            "templates": [tpl.to_dict() for tpl in self._templates.values()]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved {len(self._templates)} templates to {path}")
        return True

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self._templates.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get template statistics."""
        return {
            "total_templates": len(self._templates),
            "role_templates": len([n for n in self._templates if n.startswith("role/")]),
            "custom_templates": len([n for n in self._templates if not n.startswith("role/")]),
        }
