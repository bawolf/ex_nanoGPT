defmodule ExNanoGPT.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children =
      if Application.get_env(:ex_nano_gpt, :start_web, false) do
        [{ExNanoGPTWeb.Endpoint, []}]
      else
        []
      end

    opts = [strategy: :one_for_one, name: ExNanoGPT.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
