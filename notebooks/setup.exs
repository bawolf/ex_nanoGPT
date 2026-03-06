defmodule NotebookSetup do
  @moduledoc false

  def platform_deps do
    case :os.type() do
      {:unix, :darwin} -> [{:emlx, github: "elixir-nx/emlx", branch: "main"}]
      _ -> []
    end
  end

  def configure_backend! do
    default = if match?({:unix, :darwin}, :os.type()), do: "emlx", else: "exla"
    backend = System.get_env("NX_BACKEND", default)

    case backend do
      "exla" ->
        Nx.global_default_backend(EXLA.Backend)
        Nx.Defn.global_default_options(compiler: EXLA)
        IO.puts("Backend: EXLA (CPU on Mac, CUDA on Linux)")

      "emlx_cpu" ->
        Nx.global_default_backend({EMLX.Backend, device: :cpu})
        Nx.Defn.global_default_options(compiler: EMLX)
        IO.puts("Backend: EMLX (CPU)")

      _ ->
        Nx.global_default_backend({EMLX.Backend, device: :gpu})
        Nx.Defn.global_default_options(compiler: EMLX)
        IO.puts("Backend: EMLX (Apple Metal GPU)")
    end

    backend
  end
end
