defmodule ExNanoGPT.OptimizerTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Optimizer

  @config %{
    learning_rate: 1.0e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1.0e-8,
    weight_decay: 0.1,
    warmup_iters: 100,
    lr_decay_iters: 1000,
    min_lr: 1.0e-4,
    grad_clip: 1.0
  }

  test "LR schedule: warmup phase is linear" do
    lr_0 = Optimizer.get_lr(0, @config)
    lr_50 = Optimizer.get_lr(50, @config)
    lr_99 = Optimizer.get_lr(99, @config)

    assert lr_0 < lr_50
    assert lr_50 < lr_99
    assert_in_delta lr_0, 1.0e-3 * 1 / 101, 1.0e-8
    assert_in_delta lr_50, 1.0e-3 * 51 / 101, 1.0e-8
  end

  test "LR schedule: peak at warmup boundary" do
    lr_at_warmup = Optimizer.get_lr(100, @config)
    assert_in_delta lr_at_warmup, 1.0e-3, 1.0e-6
  end

  test "LR schedule: cosine decay between warmup and decay_iters" do
    lr_mid = Optimizer.get_lr(550, @config)
    assert lr_mid < 1.0e-3
    assert lr_mid > 1.0e-4
  end

  test "LR schedule: returns min_lr after decay_iters" do
    lr_end = Optimizer.get_lr(2000, @config)
    assert_in_delta lr_end, 1.0e-4, 1.0e-8
  end

  test "init creates matching state structure" do
    params = %{
      weight: Nx.broadcast(1.0, {4, 8}),
      bias: Nx.broadcast(0.0, {8})
    }

    {step, state} = Optimizer.init(params)

    assert step == 0
    assert Nx.shape(state.weight.m) == {4, 8}
    assert Nx.shape(state.weight.v) == {4, 8}
    assert Nx.shape(state.bias.m) == {8}
    assert Nx.shape(state.bias.v) == {8}

    assert Nx.to_number(Nx.sum(state.weight.m)) == 0.0
    assert Nx.to_number(Nx.sum(state.weight.v)) == 0.0
  end

  test "one step moves parameters in direction of negative gradient" do
    params = %{w: Nx.broadcast(1.0, {3, 3})}
    grads = %{w: Nx.broadcast(0.1, {3, 3})}

    {_step, state} = Optimizer.init(params)
    config = %{@config | weight_decay: 0.0}

    {new_params, _new_state} = Optimizer.step(params, grads, {0, state}, config)

    # With positive gradient, params should decrease
    diff = Nx.subtract(new_params.w, params.w)
    assert Nx.to_number(Nx.reduce_max(diff)) < 0
  end

  test "weight decay only applies to 2D+ tensors" do
    params = %{
      weight_2d: Nx.broadcast(1.0, {4, 4}),
      bias_1d: Nx.broadcast(1.0, {4})
    }

    grads = %{
      weight_2d: Nx.broadcast(0.0, {4, 4}),
      bias_1d: Nx.broadcast(0.0, {4})
    }

    {_step, state} = Optimizer.init(params)
    config = %{@config | weight_decay: 0.5, learning_rate: 0.1, warmup_iters: 0}

    {new_params, _} = Optimizer.step(params, grads, {0, state}, config)

    # 2D weight should have moved (due to weight decay alone)
    weight_diff = Nx.to_number(Nx.sum(Nx.subtract(new_params.weight_2d, params.weight_2d)))
    assert weight_diff < 0

    # 1D bias should NOT have moved (zero gradient, no weight decay)
    bias_diff = Nx.to_number(Nx.sum(Nx.subtract(new_params.bias_1d, params.bias_1d)))
    assert_in_delta bias_diff, 0.0, 1.0e-6
  end

  test "gradient clipping scales down large gradients" do
    grads = %{w: Nx.broadcast(10.0, {100})}
    clipped = Optimizer.clip_grad_norm(grads, 1.0)

    norm_after = Nx.to_number(Nx.sum(Nx.multiply(clipped.w, clipped.w))) |> :math.sqrt()
    assert_in_delta norm_after, 1.0, 0.01
  end

  test "gradient clipping is no-op for small gradients" do
    grads = %{w: Nx.broadcast(0.01, {10})}
    clipped = Optimizer.clip_grad_norm(grads, 1.0)

    assert Nx.to_number(Nx.sum(Nx.subtract(clipped.w, grads.w))) == 0.0
  end

  test "multiple steps reduce loss on a trivial problem" do
    # Simple test: optimize a single weight to minimize (w - 3)^2
    params = %{w: Nx.tensor([[0.0]])}
    {_step, state} = Optimizer.init(params)

    config = %{@config | learning_rate: 0.1, weight_decay: 0.0, warmup_iters: 0, grad_clip: 0.0}

    {final_params, _} =
      Enum.reduce(0..499, {params, {0, state}}, fn _i, {params, opt_state} ->
        grad_val = Nx.multiply(2.0, Nx.subtract(params.w, Nx.tensor([[3.0]])))
        grads = %{w: grad_val}
        Optimizer.step(params, grads, opt_state, config)
      end)

    assert_in_delta Nx.to_number(final_params.w[0][0]), 3.0, 0.01
  end

  test "handles nested params with tuples" do
    params = %{
      blocks: {
        %{w: Nx.broadcast(1.0, {2, 2}), b: Nx.broadcast(0.0, {2})},
        %{w: Nx.broadcast(1.0, {2, 2}), b: Nx.broadcast(0.0, {2})}
      }
    }

    grads = %{
      blocks: {
        %{w: Nx.broadcast(0.1, {2, 2}), b: Nx.broadcast(0.1, {2})},
        %{w: Nx.broadcast(0.1, {2, 2}), b: Nx.broadcast(0.1, {2})}
      }
    }

    {_step, state} = Optimizer.init(params)
    {new_params, _} = Optimizer.step(params, grads, {0, state}, @config)

    assert Nx.shape(elem(new_params.blocks, 0).w) == {2, 2}
    assert Nx.shape(elem(new_params.blocks, 1).b) == {2}
  end
end
