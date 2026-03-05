defmodule ExNanoGPT.Sampler do
  @moduledoc """
  Text generation via autoregressive sampling.

  Mirrors nanoGPT's `GPT.generate()` (model.py) and `sample.py`:
  - Feed token sequence through model to get next-token logits
  - Scale logits by temperature
  - Optionally crop to top-k most likely tokens
  - Sample from the resulting probability distribution
  - Append sampled token and repeat
  """

  import Nx.Defn

  alias ExNanoGPT.{Data, Model}

  @typedoc """
  Sampling configuration.
  - `:max_new_tokens` - number of tokens to generate
  - `:temperature` - controls randomness (1.0 = neutral, <1 = more deterministic, >1 = more random)
  - `:top_k` - if set, only sample from the top-k most likely tokens (nil = no filtering)
  - `:block_size` - max context window (must match model's block_size)
  """
  @type sample_config :: %{
          max_new_tokens: pos_integer(),
          temperature: float(),
          top_k: pos_integer() | nil,
          block_size: pos_integer()
        }

  @doc """
  Generate text from a trained model.

  ## Inputs
    * `params` - trained model parameters
    * `model_config` - model architecture config
    * `data` - Data.t() struct with encode/decode mappings
    * `prompt` - starting text string
    * `key` - PRNG key for sampling randomness

  ## Options
    * `:max_new_tokens` - tokens to generate (default: 500)
    * `:temperature` - sampling temperature (default: 0.8)
    * `:top_k` - top-k filtering (default: 200)
  """
  @spec generate_text(Model.params(), Model.config(), Data.t(), String.t(), Nx.Tensor.t(), keyword()) :: String.t()
  def generate_text(params, model_config, data, prompt, key, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 500)
    temperature = Keyword.get(opts, :temperature, 0.8)
    top_k = Keyword.get(opts, :top_k, 200)

    prompt_ids = Data.encode(data, prompt)
    idx = Nx.tensor([prompt_ids], type: :s32)

    generated_idx = generate(idx, params, model_config, key,
      max_new_tokens: max_new_tokens,
      temperature: temperature,
      top_k: top_k
    )

    generated_idx
    |> Nx.squeeze(axes: [0])
    |> Nx.to_flat_list()
    |> Data.decode(data)
  end

  @doc """
  Autoregressive generation loop.

  Mirrors nanoGPT's `GPT.generate()`:

      for _ in range(max_new_tokens):
          idx_cond = idx[:, -block_size:]      # crop context
          logits, _ = self(idx_cond)            # forward
          logits = logits[:, -1, :] / temperature
          if top_k: ...                         # filter
          probs = softmax(logits)
          idx_next = multinomial(probs)
          idx = cat(idx, idx_next)

  ## Inputs
    * `idx` - starting token indices, shape `{batch, t}`
    * `params` - model parameters
    * `model_config` - model configuration
    * `key` - PRNG key

  ## Returns
  Extended token sequence, shape `{batch, t + max_new_tokens}`.
  """
  @spec generate(Nx.Tensor.t(), Model.params(), Model.config(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def generate(idx, params, model_config, key, opts \\ []) do
    max_new_tokens = Keyword.get(opts, :max_new_tokens, 500)
    temperature = Keyword.get(opts, :temperature, 1.0)
    top_k = Keyword.get(opts, :top_k, nil)
    block_size = model_config.block_size

    Enum.reduce(0..(max_new_tokens - 1), {idx, key}, fn _i, {idx, key} ->
      # Crop context to block_size if sequence is too long
      seq_len = Nx.axis_size(idx, 1)
      idx_cond =
        if seq_len > block_size do
          Nx.slice_along_axis(idx, seq_len - block_size, block_size, axis: 1)
        else
          idx
        end

      # Forward pass (inference mode: last position logits only)
      logits = Model.forward(idx_cond, params, model_config, key)

      # logits shape: {batch, 1, vocab_size} -> squeeze to {batch, vocab_size}
      logits = Nx.squeeze(logits, axes: [1])

      # Scale by temperature
      logits = Nx.divide(logits, temperature)

      # Top-k filtering
      logits =
        if top_k do
          apply_top_k(logits, top_k)
        else
          logits
        end

      # Softmax -> probabilities
      probs = softmax(logits)

      # Sample from the distribution
      {key, sample_key} = split_key(key)
      idx_next = sample_multinomial(probs, sample_key)

      # Append: {batch, t} cat {batch, 1} -> {batch, t+1}
      new_idx = Nx.concatenate([idx, idx_next], axis: 1)

      {new_idx, key}
    end)
    |> elem(0)
  end

  @doc """
  Apply top-k filtering to logits.

  Sets all logits below the k-th highest value to negative infinity,
  so they have zero probability after softmax.
  """
  @spec apply_top_k(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def apply_top_k(logits, k) do
    {_batch, vocab_size} = Nx.shape(logits)
    k = min(k, vocab_size)

    # Sort descending, find the k-th value as threshold
    sorted = Nx.sort(logits, axis: -1, direction: :desc)
    threshold = Nx.slice_along_axis(sorted, k - 1, 1, axis: -1)

    mask = Nx.less(logits, threshold)
    Nx.select(mask, Nx.broadcast(Nx.Constants.neg_infinity(:f32), Nx.shape(logits)), logits)
  end

  defn softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.exp(logits - max)
    shifted / Nx.sum(shifted, axes: [-1], keep_axes: true)
  end

  @doc """
  Sample one token index per batch element from a probability distribution.

  Uses the Gumbel-max trick: argmax(log(probs) + Gumbel noise)
  which is equivalent to multinomial sampling.
  """
  defn sample_multinomial(probs, key) do
    log_probs = Nx.log(Nx.max(probs, 1.0e-10))
    {uniform, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
    gumbel_noise = -Nx.log(-Nx.log(Nx.max(uniform, 1.0e-10)))
    perturbed = log_probs + gumbel_noise
    indices = Nx.argmax(perturbed, axis: -1, keep_axis: true)
    Nx.as_type(indices, :s32)
  end

  defn split_key(key) do
    keys = Nx.Random.split(key)
    {keys[0], keys[1]}
  end
end
