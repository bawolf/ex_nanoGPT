defmodule ExNanoGPT.TrainerTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.{Model, Trainer}

  @model_config %{
    vocab_size: 65,
    block_size: 8,
    n_layer: 1,
    n_head: 2,
    n_embd: 16,
    dropout: 0.0,
    bias: true
  }

  test "compute_loss_and_grads returns loss and matching gradient structure" do
    key = Nx.Random.key(42)
    params = Model.init_params(@model_config, key)

    x = Nx.tensor([[1, 5, 10, 20, 30, 15, 8, 3]])
    y = Nx.tensor([[5, 10, 20, 30, 15, 8, 3, 42]])

    dropout_key = Nx.Random.key(0)
    {loss, grads} = Trainer.compute_loss_and_grads(params, x, y, @model_config, dropout_key)

    # Loss should be a positive scalar
    assert Nx.shape(loss) == {}
    assert Nx.to_number(loss) > 0

    # Gradients should have the same structure as params
    assert Nx.shape(grads.wte) == Nx.shape(params.wte)
    assert Nx.shape(grads.wpe) == Nx.shape(params.wpe)
    assert Nx.shape(grads.ln_f.weight) == Nx.shape(params.ln_f.weight)

    block_grad = elem(grads.blocks, 0)
    block_param = elem(params.blocks, 0)
    assert Nx.shape(block_grad.attn.c_attn_weight) == Nx.shape(block_param.attn.c_attn_weight)
  end

  test "one training step reduces loss" do
    key = Nx.Random.key(42)
    params = Model.init_params(@model_config, key)

    x = Nx.tensor([[1, 5, 10, 20, 30, 15, 8, 3], [60, 2, 45, 12, 7, 55, 33, 0]])
    y = Nx.tensor([[5, 10, 20, 30, 15, 8, 3, 42], [2, 45, 12, 7, 55, 33, 0, 11]])

    dropout_key = Nx.Random.key(0)

    optim_config = %{
      ExNanoGPT.Optimizer.default_config()
      | learning_rate: 1.0e-3,
        warmup_iters: 0,
        weight_decay: 0.0,
        grad_clip: 0.0
    }

    logits_before =
      Model.forward_train(x, params, dropout_key,
        n_head: @model_config.n_head,
        dropout_rate: @model_config.dropout
      )

    loss_before = Nx.to_number(Model.cross_entropy_loss(logits_before, y))

    # Take a step
    {_loss, grads} = Trainer.compute_loss_and_grads(params, x, y, @model_config, dropout_key)
    opt_state = ExNanoGPT.Optimizer.init(params)
    {new_params, _} = ExNanoGPT.Optimizer.step(params, grads, opt_state, optim_config)

    logits_after =
      Model.forward_train(x, new_params, dropout_key,
        n_head: @model_config.n_head,
        dropout_rate: @model_config.dropout
      )

    loss_after = Nx.to_number(Model.cross_entropy_loss(logits_after, y))

    assert loss_after < loss_before
  end

  test "checkpoint round-trip preserves params" do
    key = Nx.Random.key(42)
    params = Model.init_params(@model_config, key)

    dir = Path.join(System.tmp_dir!(), "ex_nano_gpt_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(dir)

    Trainer.save_checkpoint(params, nil, @model_config, 0, :infinity, dir)
    loaded = Trainer.load_checkpoint(dir)

    assert Nx.to_number(Nx.sum(Nx.subtract(loaded.params.wte, params.wte))) == 0.0
    assert loaded.model_config == @model_config
    assert loaded.iter == 0
    assert loaded.best_val_loss == :infinity

    File.rm_rf!(dir)
  end
end
