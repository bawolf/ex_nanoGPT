defmodule ExNanoGPT.EmbeddingTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Embedding
  import ExNanoGPT.Test.GoldenHelpers

  test "token embedding matches PyTorch" do
    wte = load_golden("emb_wte")
    idx = load_golden("emb_idx") |> Nx.as_type(:s64)
    expected = load_golden("emb_tok")

    actual = Embedding.token_embedding(idx, wte)
    assert_close(actual, expected)
  end

  test "position embedding matches PyTorch" do
    wpe = load_golden("emb_wpe")
    expected = load_golden("emb_pos")

    actual = Embedding.position_embedding(wpe, seq_len: 5)
    assert_close(actual, expected)
  end

  test "combined tok + pos embedding matches PyTorch" do
    wte = load_golden("emb_wte")
    wpe = load_golden("emb_wpe")
    idx = load_golden("emb_idx") |> Nx.as_type(:s64)
    expected = load_golden("emb_combined")

    params = %{wte: wte, wpe: wpe}
    key = Nx.Random.key(0)
    actual = Embedding.forward(idx, params, key, dropout_rate: 0.0, training: false)

    assert_close(actual, expected)
  end

  test "init_params creates correct shapes" do
    key = Nx.Random.key(42)
    params = Embedding.init_params(65, 256, 384, key)

    assert Nx.shape(params.wte) == {65, 384}
    assert Nx.shape(params.wpe) == {256, 384}
  end

  test "dropout with training=true and rate > 0 zeros some values" do
    key = Nx.Random.key(42)
    params = Embedding.init_params(10, 8, 16, key)
    idx = Nx.tensor([[0, 1, 2, 3]])

    dropout_key = Nx.Random.key(99)
    out = Embedding.forward(idx, params, dropout_key, dropout_rate: 0.5, training: true)

    zeros = Nx.equal(out, 0.0) |> Nx.sum() |> Nx.to_number()
    total = Nx.size(out)
    assert zeros > 0, "Expected some zeros from dropout"
    assert zeros < total, "Expected some non-zeros from dropout"
  end
end
