defmodule ExNanoGPT.V2.TokenizerTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.Tokenizer

  @training_text "the cat sat on the mat. the cat sat on the hat. the dog sat on the log."

  describe "train/2" do
    test "creates tokenizer with correct vocab size" do
      tok = Tokenizer.train(@training_text, vocab_size: 280)
      assert tok.vocab_size == 280
    end

    test "merges reduce total token count" do
      bytes_before = byte_size(@training_text)
      tok = Tokenizer.train(@training_text, vocab_size: 280)
      ids = Tokenizer.encode(tok, @training_text)
      assert length(ids) < bytes_before
    end
  end

  describe "encode/3 and decode/2" do
    setup do
      tok = Tokenizer.train(@training_text, vocab_size: 300)
      %{tok: tok}
    end

    test "roundtrip: decode(encode(text)) == text", %{tok: tok} do
      texts = [
        "hello world",
        "the cat sat on the mat",
        @training_text,
        "abc",
        ""
      ]

      for text <- texts do
        ids = Tokenizer.encode(tok, text)
        decoded = Tokenizer.decode(tok, ids)
        assert decoded == text, "Failed roundtrip for: #{inspect(text)}"
      end
    end

    test "encodes bytes not in training text", %{tok: tok} do
      ids = Tokenizer.encode(tok, "ZZZZ")
      decoded = Tokenizer.decode(tok, ids)
      assert decoded == "ZZZZ"
    end

    test "encodes empty string", %{tok: tok} do
      assert Tokenizer.encode(tok, "") == []
      assert Tokenizer.decode(tok, []) == ""
    end

    test "handles UTF-8 multi-byte characters", %{tok: tok} do
      text = "hello 世界"
      ids = Tokenizer.encode(tok, text)
      decoded = Tokenizer.decode(tok, ids)
      assert decoded == text
    end
  end

  describe "special tokens" do
    setup do
      tok = Tokenizer.train("hello", vocab_size: 280)
      %{tok: tok}
    end

    test "special tokens get unique IDs", %{tok: tok} do
      bos = Tokenizer.encode_special(tok, "<|bos|>")
      user_start = Tokenizer.encode_special(tok, "<|user_start|>")
      assert bos != user_start
      assert bos >= 256
    end

    test "encode with prepend/append", %{tok: tok} do
      ids = Tokenizer.encode(tok, "hi", prepend: "<|bos|>")
      assert hd(ids) == Tokenizer.bos_token_id(tok)
    end

    test "decode special tokens", %{tok: tok} do
      bos_id = Tokenizer.bos_token_id(tok)
      ids = [bos_id | Tokenizer.encode(tok, "hi")]
      decoded = Tokenizer.decode(tok, ids)
      assert decoded == "<|bos|>hi"
    end
  end

  describe "persistence" do
    test "save and load roundtrip" do
      tok = Tokenizer.train(@training_text, vocab_size: 290)
      path = Path.join(System.tmp_dir!(), "test_tokenizer.etf")

      Tokenizer.save(tok, path)
      tok2 = Tokenizer.load(path)

      text = "the cat sat"
      assert Tokenizer.encode(tok, text) == Tokenizer.encode(tok2, text)
      assert tok.vocab_size == tok2.vocab_size

      File.rm(path)
    end
  end

  describe "BPE correctness" do
    test "most frequent pair is merged first" do
      # 256 base bytes + 9 special + 1 merge = 266
      tok = Tokenizer.train("aaaa", vocab_size: 266)
      ids = Tokenizer.encode(tok, "aaaa")
      # After merging (97,97)->256, we get [256,256]
      assert length(ids) == 2
    end

    test "merge order matches greedy frequency" do
      # 256 base bytes + 9 special + 2 merges = 267
      tok = Tokenizer.train("ababab", vocab_size: 267)
      ids = Tokenizer.encode(tok, "ababab")
      # First merge: (97,98)->256 "ab", giving [256, 256, 256]
      # Second merge: (256,256)->257 "abab", giving [257, 256]
      assert length(ids) == 2
    end
  end
end
