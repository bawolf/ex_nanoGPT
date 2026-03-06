defmodule ExNanoGPT.V2.ConversationTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.{Conversation, Tokenizer}

  setup do
    tok = Tokenizer.train("hello world how are you doing today", vocab_size: 280)
    %{tok: tok}
  end

  describe "render/2" do
    test "single user+assistant turn", %{tok: tok} do
      turns = [
        %{role: "user", content: "hi"},
        %{role: "assistant", content: "hello"}
      ]

      {ids, mask} = Conversation.render(turns, tok)

      assert length(ids) == length(mask)
      assert hd(ids) == Tokenizer.bos_token_id(tok)
      assert hd(mask) == 0

      # Check that some mask values are 1 (assistant tokens)
      assert Enum.any?(mask, &(&1 == 1))
      # Check that mask starts with 0s (bos + user tokens)
      assert Enum.at(mask, 0) == 0
    end

    test "loss mask is 0 for user, 1 for assistant content", %{tok: tok} do
      turns = [
        %{role: "user", content: "a"},
        %{role: "assistant", content: "b"}
      ]

      {ids, mask} = Conversation.render(turns, tok)

      # Find assistant_start token
      ast_start = Tokenizer.encode_special(tok, "<|assistant_start|>")
      ast_end = Tokenizer.encode_special(tok, "<|assistant_end|>")

      # assistant_start has mask 0, content and end have mask 1
      ast_start_idx = Enum.find_index(ids, &(&1 == ast_start))
      ast_end_idx = Enum.find_index(ids, &(&1 == ast_end))

      assert Enum.at(mask, ast_start_idx) == 0
      assert Enum.at(mask, ast_end_idx) == 1

      # All positions between start and end (exclusive of start) should be 1
      for i <- (ast_start_idx + 1)..ast_end_idx do
        assert Enum.at(mask, i) == 1, "Expected mask[#{i}]=1, got #{Enum.at(mask, i)}"
      end
    end

    test "multi-turn conversation", %{tok: tok} do
      turns = [
        %{role: "user", content: "hi"},
        %{role: "assistant", content: "hey"},
        %{role: "user", content: "how"},
        %{role: "assistant", content: "good"}
      ]

      {ids, mask} = Conversation.render(turns, tok)
      assert length(ids) == length(mask)

      ones = Enum.count(mask, &(&1 == 1))
      assert ones > 0
    end
  end

  describe "prepare_batch/3" do
    test "creates padded tensors with correct shapes", %{tok: tok} do
      convos = [
        [%{role: "user", content: "hi"}, %{role: "assistant", content: "hey"}],
        [%{role: "user", content: "yo"}, %{role: "assistant", content: "sup"}]
      ]

      {input_ids, target_ids, loss_mask} = Conversation.prepare_batch(convos, tok, 32)

      assert Nx.shape(input_ids) == {2, 32}
      assert Nx.shape(target_ids) == {2, 32}
      assert Nx.shape(loss_mask) == {2, 32}

      # Target should be input shifted by 1
      first_input = Nx.to_flat_list(input_ids[0])
      first_target = Nx.to_flat_list(target_ids[0])
      {orig_ids, _mask} = Conversation.render(hd(convos), tok)
      # input = orig[0..max_len-1], target = orig[1..max_len]
      n = min(length(orig_ids) - 1, 32)
      assert Enum.take(first_input, n) == Enum.take(orig_ids, n)
      assert Enum.take(first_target, n) == Enum.slice(orig_ids, 1, n)
    end
  end

  describe "load_jsonl/1" do
    test "parses conversation JSONL" do
      path = Path.join(System.tmp_dir!(), "test_convos.jsonl")
      content = """
      {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
      {"conversations": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "see ya"}]}
      """
      File.write!(path, content)

      convos = Conversation.load_jsonl(path)
      assert length(convos) == 2
      assert hd(hd(convos)).role == "user"
      assert hd(hd(convos)).content == "hi"

      File.rm(path)
    end
  end
end
