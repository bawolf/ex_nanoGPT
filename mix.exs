defmodule ExNanoGPT.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_nano_gpt,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9.2"},
      {:exla, "~> 0.9.2"},
      {:polaris, "~> 0.1"}
    ]
  end
end
