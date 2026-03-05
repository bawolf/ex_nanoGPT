defmodule ExNanoGPT.Data do
  @moduledoc """
  Shakespeare character-level dataset preparation.

  Mirrors nanoGPT's data/shakespeare_char/prepare.py:
  - Downloads the tiny Shakespeare dataset
  - Builds a character-level vocabulary (sorted unique chars)
  - Encodes text as uint16 integers
  - Splits 90% train / 10% val
  - Saves as binary files (little-endian uint16, matching numpy's .tofile())
  """

  @shakespeare_url "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
  @data_dir Path.join(File.cwd!(), "data")

  defstruct [:vocab_size, :stoi, :itos]

  def data_dir, do: @data_dir

  @doc """
  Run the full data preparation pipeline.
  Downloads Shakespeare, builds vocab, encodes, splits, and saves binary files.
  Returns the metadata struct.
  """
  def prepare(opts \\ []) do
    data_dir = Keyword.get(opts, :data_dir, @data_dir)
    File.mkdir_p!(data_dir)

    text = download(data_dir)
    meta = build_vocab(text)

    {train_text, val_text} = split(text, 0.9)
    train_ids = encode(meta, train_text)
    val_ids = encode(meta, val_text)

    save_bin(Path.join(data_dir, "train.bin"), train_ids)
    save_bin(Path.join(data_dir, "val.bin"), val_ids)
    save_meta(Path.join(data_dir, "meta.etf"), meta)

    IO.puts("Dataset prepared:")
    IO.puts("  vocab size: #{meta.vocab_size}")
    IO.puts("  train tokens: #{length(train_ids)}")
    IO.puts("  val tokens: #{length(val_ids)}")

    meta
  end

  @doc """
  Download the Shakespeare text file if not already present.
  """
  def download(data_dir \\ @data_dir) do
    path = Path.join(data_dir, "input.txt")

    if File.exists?(path) do
      File.read!(path)
    else
      IO.puts("Downloading Shakespeare dataset...")
      {:ok, {{_, 200, _}, _, body}} = :httpc.request(:get, {~c"#{@shakespeare_url}", []}, [], body_format: :binary)
      text = IO.iodata_to_binary(body)
      File.write!(path, text)
      text
    end
  end

  @doc """
  Build a character-level vocabulary from the text.
  Chars are sorted (matching Python's sorted(list(set(data)))).
  """
  def build_vocab(text) do
    chars =
      text
      |> String.to_charlist()
      |> Enum.uniq()
      |> Enum.sort()

    stoi = chars |> Enum.with_index() |> Map.new()
    itos = chars |> Enum.with_index() |> Map.new(fn {ch, i} -> {i, ch} end)

    %__MODULE__{
      vocab_size: length(chars),
      stoi: stoi,
      itos: itos
    }
  end

  @doc """
  Encode a string into a list of integers using the vocabulary.
  """
  def encode(%__MODULE__{stoi: stoi}, text) do
    text
    |> String.to_charlist()
    |> Enum.map(&Map.fetch!(stoi, &1))
  end

  @doc """
  Decode a list of integers back into a string.
  """
  def decode(%__MODULE__{itos: itos}, ids) do
    ids
    |> Enum.map(&Map.fetch!(itos, &1))
    |> List.to_string()
  end

  @doc """
  Split text into train/val at the given ratio.
  """
  def split(text, ratio) do
    n = String.length(text)
    split_at = trunc(n * ratio)
    {String.slice(text, 0, split_at), String.slice(text, split_at, n - split_at)}
  end

  @doc """
  Save a list of integer token IDs as a binary file (little-endian uint16).
  This matches numpy's `np.array(ids, dtype=np.uint16).tofile(path)`.
  """
  def save_bin(path, ids) do
    binary = ids |> Enum.map(&<<&1::little-unsigned-16>>) |> IO.iodata_to_binary()
    File.write!(path, binary)
  end

  @doc """
  Load a binary file of uint16 tokens into an Nx tensor.
  """
  def load_bin(path) do
    binary = File.read!(path)
    count = byte_size(binary) |> div(2)

    binary
    |> Nx.from_binary(:u16)
    |> Nx.reshape({count})
  end

  @doc """
  Save metadata as Erlang Term Format.
  """
  def save_meta(path, meta) do
    File.write!(path, :erlang.term_to_binary(meta))
  end

  @doc """
  Load metadata from Erlang Term Format.
  """
  def load_meta(path) do
    path |> File.read!() |> :erlang.binary_to_term()
  end
end
