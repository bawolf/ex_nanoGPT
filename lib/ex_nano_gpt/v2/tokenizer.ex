defmodule ExNanoGPT.V2.Tokenizer do
  @moduledoc """
  BPE (Byte-Pair Encoding) tokenizer implemented from scratch.

  Mirrors nanochat's GPT-4-style tokenizer:
  1. Start with individual bytes (256 base tokens)
  2. Iteratively merge the most frequent adjacent pair
  3. Build a merge table mapping pairs -> new token IDs
  4. Encode: apply merges greedily to byte sequences
  5. Decode: map token IDs back to bytes, decode UTF-8

  Also supports special tokens for conversation formatting.
  """

  @special_tokens ~w(<|bos|> <|user_start|> <|user_end|> <|assistant_start|> <|assistant_end|> <|python_start|> <|python_end|> <|output_start|> <|output_end|>)

  defstruct [:merges, :vocab, :vocab_inv, :special_map, :special_map_inv, :vocab_size]

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------

  @doc """
  Train a BPE tokenizer from a text corpus.

  Returns a `%Tokenizer{}` with the merge table and vocabulary.

  ## Options
    * `:vocab_size` - target vocabulary size (default: 512)
  """
  def train(text, opts \\ []) do
    target_vocab = Keyword.get(opts, :vocab_size, 512)
    base_vocab = 256
    num_special = length(@special_tokens)
    num_merges = target_vocab - base_vocab - num_special

    if num_merges < 0, do: raise("vocab_size must be >= #{base_vocab + num_special}")

    # Start with byte-level token IDs
    ids = text |> :binary.bin_to_list()
    next_id = base_vocab

    {merges, _ids, next_id} =
      Enum.reduce(1..max(num_merges, 0), {[], ids, next_id}, fn _step, {merges, ids, next_id} ->
        counts = count_pairs(ids)

        if map_size(counts) == 0 do
          {merges, ids, next_id}
        else
          {best_pair, _count} = Enum.max_by(counts, fn {_pair, count} -> count end)
          new_ids = merge_pair(ids, best_pair, next_id)
          {[{best_pair, next_id} | merges], new_ids, next_id + 1}
        end
      end)

    merges = Enum.reverse(merges)

    # Build vocab: base bytes + merge tokens + special tokens
    vocab = build_vocab(merges, next_id)
    vocab_inv = Map.new(vocab, fn {k, v} -> {v, k} end)

    # Special tokens get IDs after the last merge token
    special_start = next_id

    special_map =
      @special_tokens
      |> Enum.with_index(special_start)
      |> Map.new(fn {name, id} -> {name, id} end)

    special_map_inv = Map.new(special_map, fn {k, v} -> {v, k} end)

    %__MODULE__{
      merges: merges,
      vocab: vocab,
      vocab_inv: vocab_inv,
      special_map: special_map,
      special_map_inv: special_map_inv,
      vocab_size: special_start + num_special
    }
  end

  defp count_pairs([_]), do: %{}
  defp count_pairs([]), do: %{}

  defp count_pairs(ids) do
    ids
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce(%{}, fn [a, b], acc ->
      Map.update(acc, {a, b}, 1, &(&1 + 1))
    end)
  end

  defp merge_pair(ids, {p1, p2}, new_id) do
    do_merge(ids, p1, p2, new_id, [])
  end

  defp do_merge([], _p1, _p2, _new_id, acc), do: Enum.reverse(acc)
  defp do_merge([a], _p1, _p2, _new_id, acc), do: Enum.reverse([a | acc])

  defp do_merge([a, b | rest], p1, p2, new_id, acc) when a == p1 and b == p2 do
    do_merge(rest, p1, p2, new_id, [new_id | acc])
  end

  defp do_merge([a | rest], p1, p2, new_id, acc) do
    do_merge(rest, p1, p2, new_id, [a | acc])
  end

  defp build_vocab(merges, _next_id) do
    base = for i <- 0..255, into: %{}, do: {[i], i}

    merges
    |> Enum.reduce(base, fn {{a, b}, id}, vocab ->
      bytes_a = find_bytes(vocab, a)
      bytes_b = find_bytes(vocab, b)
      Map.put(vocab, bytes_a ++ bytes_b, id)
    end)
  end

  defp find_bytes(vocab, id) do
    Enum.find_value(vocab, fn {bytes, vid} -> if vid == id, do: bytes end)
  end

  # ---------------------------------------------------------------------------
  # Encoding
  # ---------------------------------------------------------------------------

  @doc """
  Encode text into token IDs.

  ## Options
    * `:prepend` - special token name to prepend (e.g., `"<|bos|>"`)
    * `:append` - special token name to append
  """
  def encode(%__MODULE__{} = tok, text, opts \\ []) do
    prepend = Keyword.get(opts, :prepend)
    append = Keyword.get(opts, :append)

    ids =
      if tok.merges == [] do
        encode_longest_match(text, tok.vocab)
      else
        text |> :binary.bin_to_list() |> apply_merges(tok.merges)
      end

    ids = if prepend, do: [encode_special(tok, prepend) | ids], else: ids
    ids = if append, do: ids ++ [encode_special(tok, append)], else: ids
    ids
  end

  @doc "Encode a single special token by name."
  def encode_special(%__MODULE__{} = tok, name) do
    Map.fetch!(tok.special_map, name)
  end

  defp apply_merges(ids, []), do: ids

  defp apply_merges(ids, [{pair, new_id} | rest]) do
    {p1, p2} = pair
    ids = merge_pair(ids, {p1, p2}, new_id)
    apply_merges(ids, rest)
  end

  defp encode_longest_match(text, vocab) do
    bytes = :binary.bin_to_list(text)
    max_token_len = vocab |> Map.keys() |> Enum.map(&length/1) |> Enum.max(fn -> 1 end)
    do_longest_match(bytes, vocab, max_token_len, [])
  end

  defp do_longest_match([], _vocab, _max_len, acc), do: Enum.reverse(acc)

  defp do_longest_match(bytes, vocab, max_len, acc) do
    window = min(max_len, length(bytes))

    {token_id, consumed} =
      Enum.reduce_while(window..1//-1, nil, fn len, _acc ->
        candidate = Enum.take(bytes, len)

        case Map.get(vocab, candidate) do
          nil -> {:cont, nil}
          id -> {:halt, {id, len}}
        end
      end)
      |> case do
        nil -> {hd(bytes), 1}
        found -> found
      end

    do_longest_match(Enum.drop(bytes, consumed), vocab, max_len, [token_id | acc])
  end

  # ---------------------------------------------------------------------------
  # Decoding
  # ---------------------------------------------------------------------------

  @doc "Decode token IDs back to a string."
  def decode(%__MODULE__{} = tok, ids) do
    ids
    |> Enum.flat_map(fn id ->
      cond do
        Map.has_key?(tok.special_map_inv, id) ->
          tok.special_map_inv[id] |> :binary.bin_to_list()

        Map.has_key?(tok.vocab_inv, id) ->
          tok.vocab_inv[id]

        id < 256 ->
          [id]

        true ->
          []
      end
    end)
    |> :binary.list_to_bin()
  end

  # ---------------------------------------------------------------------------
  # Persistence
  # ---------------------------------------------------------------------------

  @doc "Save tokenizer to disk as Erlang Term Format."
  def save(%__MODULE__{} = tok, path) do
    data = %{
      merges: tok.merges,
      special_tokens: @special_tokens,
      vocab_size: tok.vocab_size
    }

    File.write!(path, :erlang.term_to_binary(data))
  end

  @doc "Load tokenizer from a JSON vocab file (from convert_tokenizer.py)."
  def load_vocab_json(path) do
    data = path |> File.read!() |> Jason.decode!()

    vocab_map = data["vocab"]

    vocab_inv =
      vocab_map
      |> Enum.map(fn {id_str, bytes} -> {String.to_integer(id_str), bytes} end)
      |> Map.new()

    vocab =
      vocab_inv
      |> Enum.map(fn {id, bytes} -> {bytes, id} end)
      |> Map.new()

    special_map = data["special_tokens"]
    special_map_inv = Map.new(special_map, fn {k, v} -> {v, k} end)

    %__MODULE__{
      merges: [],
      vocab: vocab,
      vocab_inv: vocab_inv,
      special_map: special_map,
      special_map_inv: special_map_inv,
      vocab_size: data["vocab_size"]
    }
  end

  @doc "Load tokenizer from an ETF file."
  def load(path) do
    data = path |> File.read!() |> :erlang.binary_to_term()
    merges = data.merges

    next_id =
      if merges == [] do
        256
      else
        {_pair, last_id} = List.last(merges)
        last_id + 1
      end

    vocab = build_vocab(merges, next_id)
    vocab_inv = Map.new(vocab, fn {k, v} -> {v, k} end)

    special_start = next_id

    special_map =
      data.special_tokens
      |> Enum.with_index(special_start)
      |> Map.new(fn {name, id} -> {name, id} end)

    special_map_inv = Map.new(special_map, fn {k, v} -> {v, k} end)

    %__MODULE__{
      merges: merges,
      vocab: vocab,
      vocab_inv: vocab_inv,
      special_map: special_map,
      special_map_inv: special_map_inv,
      vocab_size: data.vocab_size
    }
  end

  @doc "Return the BOS token ID."
  def bos_token_id(%__MODULE__{} = tok), do: encode_special(tok, "<|bos|>")
end
