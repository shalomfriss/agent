"""Discover and switch the active model without changing conversation state."""

from __future__ import annotations

from coding_agent.commands.base import (
    CommandContext,
    CommandOutcome,
    ExitRequested,
    SlashCommand,
    WorkflowCancelled,
)


MODEL_PROVIDERS = ("openai", "anthropic", "mlx")


def parse_model_selection(
    selection: str,
    current_provider: str,
) -> tuple[str, str]:
    """Resolve ``MODEL`` or ``PROVIDER:MODEL`` into validated components."""

    value = selection.strip()
    if not value:
        raise ValueError("Model ID cannot be empty.")

    provider = current_provider
    model = value
    prefix, separator, remainder = value.partition(":")
    if separator and prefix.lower() in MODEL_PROVIDERS:
        provider = prefix.lower()
        model = remainder.strip()
    if provider not in MODEL_PROVIDERS:
        raise ValueError(f"Unknown model provider: {provider}")
    if not model:
        raise ValueError("Model ID cannot be empty.")
    return provider, model


class ModelCommand(SlashCommand):
    """Run the model picker or switch directly from a command argument."""

    name = "/model"
    usage = "/model [PROVIDER:]MODEL"
    description = "Choose a model"
    takes_argument = True

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Switch directly or guide the user through provider and model steps."""

        if argument:
            self._switch(context, argument)
            return CommandOutcome.CONTINUE
        self._choose(context)
        return CommandOutcome.CONTINUE

    def _choose(self, context: CommandContext) -> None:
        """Prompt for a provider and one of its discovered model IDs."""

        current_provider = context.display.identity.provider
        try:
            provider = context.read_choice(
                "provider",
                default=current_provider,
                choices=MODEL_PROVIDERS,
            ).lower()
            if provider not in MODEL_PROVIDERS:
                raise ValueError(f"Unknown model provider: {provider}")

            default_model = self._default_model(context, provider)
            available_models = context.model_lister(provider)
            model_choices = tuple(
                dict.fromkeys(
                    model
                    for model in (default_model, *available_models)
                    if model
                )
            )
            model = context.read_choice(
                "model",
                default=(
                    default_model
                    or (model_choices[0] if model_choices else "")
                ),
                choices=model_choices,
            )
            self._switch(context, f"{provider}:{model}")
        except WorkflowCancelled:
            context.display.warning("Model selection cancelled")
        except KeyboardInterrupt:
            if context.confirm_exit():
                raise ExitRequested
            context.display.warning("Exit cancelled")
        except EOFError:
            context.display.warning("Model selection cancelled")
        except ValueError as error:
            context.display.warning(str(error))

    @staticmethod
    def _switch(context: CommandContext, selection: str) -> None:
        """Build a replacement runtime and update UI state only on success."""

        try:
            provider, model = parse_model_selection(
                selection,
                context.display.identity.provider,
            )
            context.model_switcher(provider, model)
        except Exception as error:
            context.display.warning(str(error))
            return
        context.display.set_model(provider, model)

    @staticmethod
    def _default_model(context: CommandContext, provider: str) -> str:
        """Return the active model or a provider-specific initial value."""

        if provider == context.display.identity.provider:
            return context.display.identity.model
        provider_defaults = context.defaults.providers[provider]
        return provider_defaults.model or ""
