defmodule ExNanoGPT.V2.ModelTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.Model
  alias ExNanoGPT.Npy
  alias ExNanoGPT.Test.GoldenHelpers

  @golden_dir Path.join([File.cwd!(), "test", "support", "golden_v2"])

  defp load(name), do: Npy.load!(Path.join(@golden_dir, "#{name}.npy"))

  describe "rms_norm/1" do
    @tag :golden
    test "matches PyTorch F.rms_norm" do
      input = load("rmsnorm_input")
      expected = load("rmsnorm_output")
      actual = Model.rms_norm(input)
      GoldenHelpers.assert_close(actual, expected, atol: 1.0e-5)
    end
  end

  describe "precompute_rope/2 + apply_rope/3" do
    @tag :golden
    test "matches nanochat RoPE implementation" do
      cos = load("rope_cos")
      sin = load("rope_sin")
      input = load("rope_input")
      expected = load("rope_output")

      {our_cos, our_sin} = Model.precompute_rope(8, 16)
      GoldenHelpers.assert_close(our_cos, cos, atol: 1.0e-6)
      GoldenHelpers.assert_close(our_sin, sin, atol: 1.0e-6)

      actual = Model.apply_rope(input, cos, sin)
      GoldenHelpers.assert_close(actual, expected, atol: 1.0e-5)
    end
  end

  describe "attention_forward/7" do
    @tag :golden
    test "GQA attention with QK norm and sliding window matches Python" do
      x = load("att_input")
      c_q = load("att_c_q_w")
      c_k = load("att_c_k_w")
      c_v = load("att_c_v_w")
      c_proj = load("att_c_proj_w")
      expected = load("att_output")

      n_head = 4
      n_kv_head = 2
      head_dim = 16
      seq_len = 8
      window_size = 4

      block_params = %{
        c_q: c_q,
        c_k: c_k,
        c_v: c_v,
        c_proj: c_proj,
        ve_gate: Nx.broadcast(Nx.tensor(0.0, type: :f32), {32, n_kv_head}),
        c_fc: Nx.broadcast(Nx.tensor(0.0, type: :f32), {64, 256}),
        c_proj_mlp: Nx.broadcast(Nx.tensor(0.0, type: :f32), {256, 64})
      }

      {cos, sin} = Model.precompute_rope(seq_len, head_dim)

      # No value embedding: pass zero tensor for v_extra
      v_extra = Nx.broadcast(Nx.tensor(0.0, type: :f32), {2, seq_len, n_kv_head, head_dim})

      actual =
        Model.attention_forward(x, block_params, v_extra, cos, sin,
          n_head: n_head,
          n_kv_head: n_kv_head,
          window_size: window_size
        )

      GoldenHelpers.assert_close(actual, expected, atol: 1.0e-4)
    end
  end

  describe "mlp_forward/2" do
    @tag :golden
    test "ReLU² MLP matches Python" do
      x = load("mlp_input")
      c_fc = load("mlp_c_fc_w")
      c_proj_mlp = load("mlp_c_proj_w")
      expected = load("mlp_output")

      block_params = %{c_fc: c_fc, c_proj_mlp: c_proj_mlp}
      actual = Model.mlp_forward(x, block_params)

      GoldenHelpers.assert_close(actual, expected, atol: 1.0e-5)
    end
  end

  describe "forward/3" do
    @tag :golden
    test "full forward pass matches Python (tiny config)" do
      idx = load("fwd_idx") |> Nx.as_type(:s64)
      wte = load("fwd_wte")
      lm_head = load("fwd_lm_head_w")
      resid_lambdas = load("fwd_resid_lambdas")
      x0_lambdas = load("fwd_x0_lambdas")
      expected_logits = load("fwd_logits")

      config = %Model{
        sequence_len: 16,
        vocab_size: 256,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 64,
        window_pattern: "SL"
      }

      blocks =
        for i <- 0..1 do
          %{
            c_q: load("fwd_block#{i}_c_q"),
            c_k: load("fwd_block#{i}_c_k"),
            c_v: load("fwd_block#{i}_c_v"),
            c_proj: load("fwd_block#{i}_c_proj"),
            ve_gate: load("fwd_block#{i}_ve_gate"),
            c_fc: load("fwd_block#{i}_c_fc"),
            c_proj_mlp: load("fwd_block#{i}_c_proj_mlp")
          }
        end
        |> List.to_tuple()

      # has_ve?(0, 2) => false, has_ve?(1, 2) => true
      ve_1 = load("fwd_ve_1")
      value_embeds = {:none, ve_1}

      params = %{
        wte: wte,
        lm_head: lm_head,
        resid_lambdas: resid_lambdas,
        x0_lambdas: x0_lambdas,
        blocks: blocks,
        value_embeds: value_embeds
      }

      actual = Model.forward(idx, params, config)

      GoldenHelpers.assert_close(actual, expected_logits, atol: 1.0e-3, rtol: 1.0e-3)
    end
  end

  describe "init_params/2" do
    test "creates params with correct shapes" do
      config = Model.tiny_config()
      params = Model.init_params(config, Nx.Random.key(0))

      assert Nx.shape(params.wte) == {256, 64}
      assert Nx.shape(params.lm_head) == {256, 64}
      assert Nx.shape(params.resid_lambdas) == {2}
      assert Nx.shape(params.x0_lambdas) == {2}
      assert tuple_size(params.blocks) == 2
      assert tuple_size(params.value_embeds) == 2

      block0 = elem(params.blocks, 0)
      assert Nx.shape(block0.c_q) == {64, 64}
      assert Nx.shape(block0.c_k) == {64, 32}
      assert Nx.shape(block0.c_v) == {64, 32}
      assert Nx.shape(block0.c_proj) == {64, 64}
      assert Nx.shape(block0.c_fc) == {64, 256}
      assert Nx.shape(block0.c_proj_mlp) == {256, 64}
    end
  end

  describe "has_ve?/2" do
    test "alternating pattern, last layer always included" do
      assert Model.has_ve?(1, 2) == true
      assert Model.has_ve?(0, 2) == false
      assert Model.has_ve?(0, 3) == true
      assert Model.has_ve?(1, 3) == false
      assert Model.has_ve?(2, 3) == true
    end
  end

  describe "compute_window_sizes/1" do
    test "computes correct window pattern" do
      config = %Model{
        sequence_len: 16,
        vocab_size: 256,
        n_layer: 4,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 64,
        window_pattern: "SL"
      }

      assert Model.compute_window_sizes(config) == [8, 16, 8, 16]
    end

    test "final layer always gets full context" do
      config = %Model{
        sequence_len: 16,
        vocab_size: 256,
        n_layer: 3,
        n_head: 4,
        n_kv_head: 2,
        n_embd: 64,
        window_pattern: "S"
      }

      assert Model.compute_window_sizes(config) == [8, 8, 16]
    end
  end

  describe "cross_entropy_loss/3" do
    test "basic loss computation" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, 1]])

      loss = Model.cross_entropy_loss(logits, targets) |> Nx.to_number()
      assert loss > 0.0 and loss < 5.0
    end

    test "ignores target=-1" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, -1]])

      loss = Model.cross_entropy_loss(logits, targets) |> Nx.to_number()
      assert loss > 0.0
    end
  end
end
