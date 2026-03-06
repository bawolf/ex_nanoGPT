defmodule ExNanoGPT.V2.Conversation do
  @moduledoc """
  Conversation rendering for SFT (Supervised Fine-Tuning).

  Converts multi-turn conversations into token sequences with loss masks.
  Mirrors nanochat's conversation format:

      <|bos|><|user_start|>user message<|user_end|>
      <|assistant_start|>assistant response<|assistant_end|>

  During SFT, loss is computed only on assistant tokens (the model learns
  to generate responses, not to predict user messages). The loss mask is
  1 for assistant tokens and 0 for everything else.
  """

  alias ExNanoGPT.V2.Tokenizer

  @type turn :: %{role: String.t(), content: String.t()}

  @doc """
  Render a conversation into token IDs and a loss mask.

  ## Arguments
    * `turns` - list of `%{role: "user" | "assistant", content: "..."}` maps
    * `tokenizer` - a trained `%Tokenizer{}`

  ## Returns
  `{token_ids, loss_mask}` where both are lists of integers.
  `loss_mask[i]` is 1 if position `i` should contribute to the loss, 0 otherwise.
  """
  def render(turns, %Tokenizer{} = tok) do
    bos = Tokenizer.encode_special(tok, "<|bos|>")

    {all_ids, all_mask} =
      turns
      |> Enum.reduce({[bos], [0]}, fn turn, {ids_acc, mask_acc} ->
        render_turn(turn, tok, ids_acc, mask_acc)
      end)

    {Enum.reverse(all_ids), Enum.reverse(all_mask)}
  end

  defp render_turn(%{role: "user", content: content}, tok, ids_acc, mask_acc) do
    start_tok = Tokenizer.encode_special(tok, "<|user_start|>")
    end_tok = Tokenizer.encode_special(tok, "<|user_end|>")
    content_ids = Tokenizer.encode(tok, content)

    new_ids = [start_tok | content_ids] ++ [end_tok]
    new_mask = List.duplicate(0, length(new_ids))

    {Enum.reverse(new_ids) ++ ids_acc, Enum.reverse(new_mask) ++ mask_acc}
  end

  defp render_turn(%{role: "assistant", content: content}, tok, ids_acc, mask_acc) do
    start_tok = Tokenizer.encode_special(tok, "<|assistant_start|>")
    end_tok = Tokenizer.encode_special(tok, "<|assistant_end|>")
    content_ids = Tokenizer.encode(tok, content)

    new_ids = [start_tok | content_ids] ++ [end_tok]
    # Loss on assistant content tokens + end token, not start token
    new_mask = [0 | List.duplicate(1, length(content_ids))] ++ [1]

    {Enum.reverse(new_ids) ++ ids_acc, Enum.reverse(new_mask) ++ mask_acc}
  end

  @doc """
  Prepare a batch of conversations for training.

  Pads/truncates to `max_len`, builds input/target pairs with shifted targets,
  and produces loss masks.

  Returns `{input_ids, target_ids, loss_mask}` as Nx tensors.
  """
  def prepare_batch(conversations, %Tokenizer{} = tok, max_len) do
    rendered =
      conversations
      |> Enum.map(fn turns ->
        {ids, mask} = render(turns, tok)
        ids = Enum.take(ids, max_len + 1)
        mask = Enum.take(mask, max_len + 1)
        pad_len = max(max_len + 1 - length(ids), 0)
        ids = ids ++ List.duplicate(0, pad_len)
        mask = mask ++ List.duplicate(0, pad_len)
        {ids, mask}
      end)

    input_ids =
      rendered
      |> Enum.map(fn {ids, _mask} -> Enum.take(ids, max_len) end)
      |> Nx.tensor(type: :s64)

    target_ids =
      rendered
      |> Enum.map(fn {ids, _mask} -> Enum.drop(ids, 1) |> Enum.take(max_len) end)
      |> Nx.tensor(type: :s64)

    loss_mask =
      rendered
      |> Enum.map(fn {_ids, mask} -> Enum.drop(mask, 1) |> Enum.take(max_len) end)
      |> Nx.tensor(type: :f32)

    {input_ids, target_ids, loss_mask}
  end

  @doc """
  Load conversations from a JSONL file.

  Each line is a JSON object with a "conversations" key containing
  a list of `{"role": ..., "content": ...}` turns.
  """
  def load_jsonl(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim/1)
    |> Stream.reject(&(&1 == ""))
    |> Enum.map(fn line ->
      data = Jason.decode!(line)

      data["conversations"]
      |> Enum.map(fn turn ->
        %{role: turn["role"], content: turn["content"]}
      end)
    end)
  end
end
