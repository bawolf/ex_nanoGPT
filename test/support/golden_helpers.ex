defmodule ExNanoGPT.Test.GoldenHelpers do
  @moduledoc """
  Helpers for golden tests: load reference .npy files and compare tensors.
  """

  alias ExNanoGPT.Test.Npy

  @golden_dir Path.join([File.cwd!(), "test", "golden"])

  @doc """
  Load a golden reference tensor by name.
  """
  def load_golden(name) do
    path = Path.join(@golden_dir, "#{name}.npy")

    if File.exists?(path) do
      Npy.load!(path)
    else
      raise "Golden test file not found: #{path}. Run: nanoGPT_ref/.venv/bin/python test/golden/extract.py"
    end
  end

  @doc """
  Assert two tensors are approximately equal within a tolerance.
  """
  def assert_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-5)

    actual_f = Nx.as_type(actual, :f32)
    expected_f = Nx.as_type(expected, :f32)

    diff = Nx.abs(Nx.subtract(actual_f, expected_f))
    threshold = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected_f)))
    within_tolerance = Nx.less_equal(diff, threshold)
    all_close = Nx.all(within_tolerance) |> Nx.to_number()

    if all_close != 1 do
      max_diff = Nx.reduce_max(diff) |> Nx.to_number()
      mean_diff = Nx.mean(diff) |> Nx.to_number()

      raise """
      Tensors are not close enough!
        max diff: #{max_diff}
        mean diff: #{mean_diff}
        atol: #{atol}, rtol: #{rtol}
        shapes: actual=#{inspect(Nx.shape(actual))}, expected=#{inspect(Nx.shape(expected))}
      """
    end

    :ok
  end
end
