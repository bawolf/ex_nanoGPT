defmodule ExNanoGPT.Optimizer do
  @moduledoc """
  AdamW optimizer with cosine learning rate schedule.

  Mirrors nanoGPT's training setup:
  - AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2017)
  - Cosine LR schedule with linear warmup
  - Selective weight decay: only 2D tensors (weights in matmuls + embeddings),
    not biases or layer norm parameters

  ## AdamW Algorithm

  For each parameter θ with gradient g at step t:

      m_t = β₁ · m_{t-1} + (1 - β₁) · g          # first moment estimate
      v_t = β₂ · v_{t-1} + (1 - β₂) · g²         # second moment estimate
      m̂_t = m_t / (1 - β₁ᵗ)                       # bias-corrected first moment
      v̂_t = v_t / (1 - β₂ᵗ)                       # bias-corrected second moment
      θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε) - lr · λ · θ_{t-1}

  The key difference from Adam: weight decay (λ · θ) is applied directly to
  the parameters, not through the gradient. This "decouples" weight decay from
  the adaptive learning rate, which works better empirically.
  """

  @typedoc """
  AdamW hyperparameters.
  - `:learning_rate` - peak learning rate (default: 6e-4)
  - `:beta1` - first moment decay (default: 0.9)
  - `:beta2` - second moment decay (default: 0.95)
  - `:eps` - numerical stability term (default: 1e-8)
  - `:weight_decay` - L2 regularization strength (default: 0.1)
  - `:warmup_iters` - linear warmup steps (default: 2000)
  - `:lr_decay_iters` - steps for cosine decay (default: 600000)
  - `:min_lr` - minimum learning rate after decay (default: 6e-5)
  - `:grad_clip` - max gradient norm, 0.0 to disable (default: 1.0)
  """
  @type config :: %{
          learning_rate: float(),
          beta1: float(),
          beta2: float(),
          eps: float(),
          weight_decay: float(),
          warmup_iters: non_neg_integer(),
          lr_decay_iters: pos_integer(),
          min_lr: float(),
          grad_clip: float()
        }

  @typedoc "Per-tensor optimizer state: first moment (m) and second moment (v)"
  @type tensor_state :: %{m: Nx.Tensor.t(), v: Nx.Tensor.t()}

  @default_config %{
    learning_rate: 6.0e-4,
    beta1: 0.9,
    beta2: 0.95,
    eps: 1.0e-8,
    weight_decay: 0.1,
    warmup_iters: 2000,
    lr_decay_iters: 600_000,
    min_lr: 6.0e-5,
    grad_clip: 1.0
  }

  @spec default_config() :: config()
  def default_config, do: @default_config

  @doc """
  Initialize optimizer state for all parameters.

  Creates zero-valued first moment (m) and second moment (v) tensors
  matching the shape and type of each parameter, plus a step counter.
  """
  @spec init(map()) :: {non_neg_integer(), map()}
  def init(params) do
    state = init_state(params)
    {0, state}
  end

  @doc """
  Compute the learning rate for a given iteration using cosine decay with warmup.

  Matches nanoGPT's `get_lr()`:
  1. Linear warmup for `warmup_iters` steps
  2. Cosine decay from `learning_rate` to `min_lr`
  3. Constant `min_lr` after `lr_decay_iters`
  """
  @spec get_lr(non_neg_integer(), config()) :: float()
  def get_lr(iter, config) do
    %{learning_rate: lr, warmup_iters: warmup, lr_decay_iters: decay_iters, min_lr: min_lr} = config

    cond do
      iter < warmup ->
        lr * (iter + 1) / (warmup + 1)

      iter > decay_iters ->
        min_lr

      true ->
        decay_ratio = (iter - warmup) / (decay_iters - warmup)
        coeff = 0.5 * (1.0 + :math.cos(:math.pi() * decay_ratio))
        min_lr + coeff * (lr - min_lr)
    end
  end

  @doc """
  Perform one AdamW update step.

  ## Inputs
    * `params` - current model parameters
    * `grads` - gradients (same structure as params)
    * `{step, state}` - step count and optimizer state from `init/1`
    * `config` - optimizer config

  ## Returns
  `{new_params, {new_step, new_state}}`.
  """
  @spec step(map(), map(), {non_neg_integer(), map()}, config()) ::
          {map(), {non_neg_integer(), map()}}
  def step(params, grads, {step_count, state}, config) do
    lr = get_lr(step_count, config)
    grads = clip_grad_norm(grads, config.grad_clip)

    # step_count is 0-indexed, but bias correction uses 1-indexed t
    t = step_count + 1

    {new_params, new_state} = adamw_update(params, grads, state, lr, t, config)

    {new_params, {t, new_state}}
  end

  @doc """
  Clip gradients by global L2 norm.

  If the total norm exceeds `max_norm`, all gradients are scaled down
  proportionally. Disabled when `max_norm` is 0.0.
  """
  @spec clip_grad_norm(map(), float()) :: map()
  def clip_grad_norm(grads, max_norm) when max_norm == 0.0, do: grads

  def clip_grad_norm(grads, max_norm) do
    total_norm = global_norm(grads)

    if total_norm > max_norm do
      scale = max_norm / (total_norm + 1.0e-6)
      scale_grads(grads, scale)
    else
      grads
    end
  end

  # --- Tree traversal for init/update/norm ---

  defp init_state(%Nx.Tensor{} = t) do
    zero = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(t)), Nx.shape(t))
    %{m: zero, v: zero}
  end

  defp init_state(map) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, init_state(v)} end)
  end

  defp init_state(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&init_state/1)
    |> List.to_tuple()
  end

  defp adamw_update(%Nx.Tensor{} = param, %Nx.Tensor{} = grad, %{m: m, v: v}, lr, t, config) do
    %{beta1: b1, beta2: b2, eps: eps, weight_decay: wd} = config

    # Only apply weight decay to 2D+ tensors (weights, not biases/norms)
    # Matches nanoGPT's configure_optimizers: p.dim() >= 2
    decay = if tuple_size(Nx.shape(param)) >= 2, do: wd, else: 0.0

    # m_t = β₁·m + (1-β₁)·g
    new_m = Nx.add(Nx.multiply(b1, m), Nx.multiply(1.0 - b1, grad))

    # v_t = β₂·v + (1-β₂)·g²
    new_v = Nx.add(Nx.multiply(b2, v), Nx.multiply(1.0 - b2, Nx.multiply(grad, grad)))

    # Bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
    m_hat = Nx.divide(new_m, 1.0 - :math.pow(b1, t))
    v_hat = Nx.divide(new_v, 1.0 - :math.pow(b2, t))

    # Adam update: lr · m̂ / (√v̂ + ε)
    adam_update = Nx.multiply(lr, Nx.divide(m_hat, Nx.add(Nx.sqrt(v_hat), eps)))

    # Decoupled weight decay: lr · λ · θ
    decay_update = Nx.multiply(lr * decay, param)

    new_param = Nx.subtract(Nx.subtract(param, adam_update), decay_update)

    {new_param, %{m: new_m, v: new_v}}
  end

  defp adamw_update(params, grads, state, lr, t, config) when is_map(params) and not is_struct(params) do
    {new_params_list, new_state_list} =
      params
      |> Map.keys()
      |> Enum.map(fn k ->
        {new_p, new_s} = adamw_update(params[k], grads[k], state[k], lr, t, config)
        {{k, new_p}, {k, new_s}}
      end)
      |> Enum.unzip()

    {Map.new(new_params_list), Map.new(new_state_list)}
  end

  defp adamw_update(params, grads, state, lr, t, config) when is_tuple(params) do
    results =
      [Tuple.to_list(params), Tuple.to_list(grads), Tuple.to_list(state)]
      |> Enum.zip()
      |> Enum.map(fn {p, g, s} -> adamw_update(p, g, s, lr, t, config) end)

    new_params = results |> Enum.map(&elem(&1, 0)) |> List.to_tuple()
    new_state = results |> Enum.map(&elem(&1, 1)) |> List.to_tuple()
    {new_params, new_state}
  end

  defp global_norm(grads) do
    sum_sq = sum_of_squares(grads)
    :math.sqrt(Nx.to_number(sum_sq))
  end

  defp sum_of_squares(%Nx.Tensor{} = t) do
    Nx.sum(Nx.multiply(t, t))
  end

  defp sum_of_squares(map) when is_map(map) and not is_struct(map) do
    map |> Map.values() |> Enum.map(&sum_of_squares/1) |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
  end

  defp sum_of_squares(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&sum_of_squares/1) |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)
  end

  defp scale_grads(%Nx.Tensor{} = t, scale) do
    Nx.multiply(t, scale)
  end

  defp scale_grads(map, scale) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, scale_grads(v, scale)} end)
  end

  defp scale_grads(tuple, scale) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&scale_grads(&1, scale)) |> List.to_tuple()
  end
end
