defmodule ExNanoGPT.V2.WeightLoaderTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.{Model, WeightLoader}

  @moduletag :golden

  @fake_ckpt_dir Path.join([File.cwd!(), "test", "support", "golden_v2", "fake_ckpt"])

  describe "load/2" do
    test "loads params with correct shapes from fake checkpoint" do
      {params, config} = WeightLoader.load(@fake_ckpt_dir)

      assert config.n_layer == 2
      assert config.n_head == 4
      assert config.n_kv_head == 2
      assert config.n_embd == 32
      assert config.vocab_size == 64

      assert Nx.shape(params.wte) == {64, 32}
      assert Nx.shape(params.lm_head) == {64, 32}
      assert Nx.shape(params.resid_lambdas) == {2}
      assert Nx.shape(params.x0_lambdas) == {2}

      assert tuple_size(params.blocks) == 2
      block0 = elem(params.blocks, 0)
      # Loaded and transposed: PyTorch (out, in) -> Elixir (in, out)
      assert Nx.shape(block0.c_q) == {32, 32}
      assert Nx.shape(block0.c_k) == {32, 16}
      assert Nx.shape(block0.c_v) == {32, 16}
      assert Nx.shape(block0.c_proj) == {32, 32}
      assert Nx.shape(block0.c_fc) == {32, 128}
      assert Nx.shape(block0.c_proj_mlp) == {128, 32}
    end

    test "value embeds loaded for correct layers" do
      {params, _config} = WeightLoader.load(@fake_ckpt_dir)

      assert elem(params.value_embeds, 0) == :none
      ve1 = elem(params.value_embeds, 1)
      assert Nx.shape(ve1) == {64, 16}
    end

    test "loaded params can run forward pass" do
      {params, config} = WeightLoader.load(@fake_ckpt_dir)
      idx = Nx.tensor([[1, 2, 3, 4]])

      logits = Model.forward(idx, params, config)
      assert Nx.shape(logits) == {1, 4, 64}

      max_logit = Nx.reduce_max(logits) |> Nx.to_number()
      assert max_logit <= 20.0
    end

    test "dtype option converts weights" do
      {params, _config} = WeightLoader.load(@fake_ckpt_dir, dtype: :f32)
      assert Nx.type(params.wte) == {:f, 32}
    end
  end
end
