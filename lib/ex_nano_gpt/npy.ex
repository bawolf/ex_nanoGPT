defmodule ExNanoGPT.Npy do
  @moduledoc """
  Minimal NumPy .npy file reader.

  Supports float32, float64, int32, int64, uint16 arrays.
  See: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
  """

  @doc """
  Load a .npy file into an Nx tensor.
  """
  def load!(path) do
    binary = File.read!(path)

    # .npy files start with magic: \x93NUMPY
    <<0x93, "NUMPY", major, _minor, header_len::little-unsigned-16, rest::binary>> =
      if byte_size(binary) > 10 do
        binary
      else
        raise "Invalid .npy file: #{path}"
      end

    if major != 1, do: raise("Only .npy format v1 supported, got v#{major}")

    <<header::binary-size(header_len), data::binary>> = rest

    {descr, shape} = parse_header(header)
    {type, byte_size_per_elem} = descr_to_nx_type(descr)

    expected_bytes = Enum.reduce(shape, 1, &(&1 * &2)) * byte_size_per_elem

    if byte_size(data) < expected_bytes do
      raise "Not enough data: expected #{expected_bytes} bytes, got #{byte_size(data)}"
    end

    data_slice = binary_part(data, 0, expected_bytes)

    data_slice
    |> Nx.from_binary(type)
    |> Nx.reshape(List.to_tuple(shape))
  end

  defp parse_header(header) do
    header = String.trim(header)

    descr =
      case Regex.run(~r/'descr'\s*:\s*'([^']+)'/, header) do
        [_, descr] -> descr
        _ -> raise "Could not parse descr from header: #{header}"
      end

    shape =
      case Regex.run(~r/'shape'\s*:\s*\(([^)]*)\)/, header) do
        [_, shape_str] ->
          shape_str
          |> String.split(",", trim: true)
          |> Enum.map(&(&1 |> String.trim() |> String.to_integer()))

        _ ->
          raise "Could not parse shape from header: #{header}"
      end

    fortran_order =
      case Regex.run(~r/'fortran_order'\s*:\s*(True|False)/, header) do
        [_, "False"] -> false
        [_, "True"] -> true
        _ -> false
      end

    if fortran_order, do: raise("Fortran order not supported")

    {descr, shape}
  end

  defp descr_to_nx_type("<f4"), do: {:f32, 4}
  defp descr_to_nx_type("<f8"), do: {:f64, 8}
  defp descr_to_nx_type("<i4"), do: {:s32, 4}
  defp descr_to_nx_type("<i8"), do: {:s64, 8}
  defp descr_to_nx_type("<u2"), do: {:u16, 2}
  defp descr_to_nx_type(other), do: raise("Unsupported dtype: #{other}")
end
