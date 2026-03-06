defmodule ExNanoGPT.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ExNanoGPTWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: ExNanoGPT.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
