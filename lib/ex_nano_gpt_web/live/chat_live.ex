defmodule ExNanoGPTWeb.ChatLive do
  use Phoenix.LiveView

  alias ExNanoGPT.V2.{Model, KVCache, Tokenizer}

  @max_tokens 256

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     assign(socket,
       messages: [],
       input: "",
       generating: false,
       temperature: 0.7,
       top_k: 40,
       model_loaded: false,
       weights_path: "",
       status: "No model loaded. Enter a checkpoint path below."
     )}
  end

  @impl true
  def handle_event("send", %{"message" => message}, socket) when message != "" do
    messages = socket.assigns.messages ++ [%{role: "user", content: message}]

    socket =
      assign(socket, messages: messages, input: "", generating: true, status: "Generating...")

    socket =
      if socket.assigns.model_loaded do
        send(self(), {:generate, message})
        socket
      else
        assign(socket,
          messages:
            messages ++ [%{role: "assistant", content: "(No model loaded -- load weights first)"}],
          generating: false,
          status: "Ready"
        )
      end

    {:noreply, socket}
  end

  def handle_event("send", _params, socket), do: {:noreply, socket}

  @impl true
  def handle_event("update_input", %{"value" => value}, socket) do
    {:noreply, assign(socket, input: value)}
  end

  @impl true
  def handle_event("set_temperature", %{"value" => value}, socket) do
    {temp, _} = Float.parse(value)
    {:noreply, assign(socket, temperature: temp)}
  end

  @impl true
  def handle_event("set_top_k", %{"value" => value}, socket) do
    {k, _} = Integer.parse(value)
    {:noreply, assign(socket, top_k: k)}
  end

  @impl true
  def handle_event("load_model", %{"path" => path}, socket) do
    socket = assign(socket, status: "Loading model from #{path}...", weights_path: path)
    send(self(), {:do_load_model, path})
    {:noreply, socket}
  end

  @impl true
  def handle_event("reset_model", _params, socket) do
    {:noreply,
     assign(socket,
       model_loaded: false,
       model_params: nil,
       model_config: nil,
       tokenizer: nil,
       messages: [],
       status: "Model unloaded. Load a new checkpoint below."
     )}
  end

  @impl true
  def handle_info({:do_load_model, path}, socket) do
    try do
      {params, config} = ExNanoGPT.V2.WeightLoader.load(path)
      tok_path = Path.join(path, "tokenizer.etf")

      tokenizer =
        if File.exists?(tok_path) do
          Tokenizer.load(tok_path)
        else
          Tokenizer.train("hello world", vocab_size: config.vocab_size)
        end

      {:noreply,
       assign(socket,
         model_params: params,
         model_config: config,
         tokenizer: tokenizer,
         model_loaded: true,
         status: "Model loaded! #{Model.count_params(params)} parameters."
       )}
    rescue
      e ->
        {:noreply, assign(socket, status: "Failed to load: #{Exception.message(e)}")}
    end
  end

  @impl true
  def handle_info({:generate, _user_msg}, socket) do
    %{
      model_params: params,
      model_config: config,
      tokenizer: tok,
      messages: messages,
      temperature: temp,
      top_k: top_k
    } = socket.assigns

    # Build token sequence from conversation
    turns =
      Enum.map(messages, fn %{role: role, content: content} ->
        %{role: role, content: content}
      end)

    {prompt_ids, _mask} = ExNanoGPT.V2.Conversation.render(turns, tok)
    ast_start = Tokenizer.encode_special(tok, "<|assistant_start|>")
    prompt_ids = prompt_ids ++ [ast_start]

    head_dim = div(config.n_embd, config.n_head)

    cache =
      KVCache.new(
        batch_size: 1,
        n_layers: config.n_layer,
        n_kv_head: config.n_kv_head,
        head_dim: head_dim,
        max_seq: config.sequence_len
      )

    # Process prompt through cache
    prompt_tensor = Nx.tensor([prompt_ids], type: :s64)
    {_logits, cache} = Model.forward_cached(prompt_tensor, params, config, cache)

    # Generate tokens
    ast_end = Tokenizer.encode_special(tok, "<|assistant_end|>")
    key = Nx.Random.key(System.system_time(:microsecond))

    generated =
      generate_tokens(params, config, tok, cache, temp, top_k, ast_end, @max_tokens, key)

    response = Tokenizer.decode(tok, generated)
    messages = socket.assigns.messages ++ [%{role: "assistant", content: response}]

    {:noreply, assign(socket, messages: messages, generating: false, status: "Ready")}
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  defp generate_tokens(_params, _config, _tok, _cache, _temp, _top_k, _end_tok, 0, _key), do: []

  defp generate_tokens(params, config, tok, cache, temp, top_k, end_tok, remaining, key) do
    # Generate one token at a time using the cache
    {logits, cache} =
      Model.forward_cached(
        Nx.tensor([[List.last(cache_last_token(cache, tok))]], type: :s64),
        params,
        config,
        cache
      )

    # Sample from logits
    logits = Nx.reshape(logits, {Nx.axis_size(logits, 2)})
    {token_id, key} = sample_token(logits, temp, top_k, key)
    token = Nx.to_number(token_id)

    if token == end_tok do
      []
    else
      [
        token
        | generate_tokens(params, config, tok, cache, temp, top_k, end_tok, remaining - 1, key)
      ]
    end
  end

  defp cache_last_token(_cache, _tok), do: [0]

  defp sample_token(logits, temperature, top_k, key) do
    logits =
      if temperature > 0 do
        Nx.divide(logits, max(temperature, 1.0e-8))
      else
        logits
      end

    # Top-K filtering
    {n} = Nx.shape(logits)
    k = min(top_k, n)

    if k < n do
      sorted = Nx.argsort(logits, direction: :desc)
      kth_idx = sorted[k - 1] |> Nx.to_number()
      threshold = logits[kth_idx]
      mask = Nx.greater_equal(logits, threshold)
      logits = Nx.select(mask, logits, Nx.Constants.neg_infinity(:f32))
      sample_categorical(logits, key)
    else
      sample_categorical(logits, key)
    end
  end

  defp sample_categorical(logits, key) do
    max = Nx.reduce_max(logits)
    probs = Nx.exp(Nx.subtract(logits, max))
    probs = Nx.divide(probs, Nx.sum(probs))
    {idx, key} = Nx.Random.choice(key, Nx.iota({Nx.axis_size(probs, 0)}), samples: 1, p: probs)
    {idx[0], key}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="flex flex-col h-screen max-w-4xl mx-auto">
      <!-- Header -->
      <header class="px-6 py-4 border-b border-gray-800">
        <div class="flex items-center justify-between">
          <h1 class="text-xl font-semibold tracking-tight">ExNanoGPT Chat</h1>
          <span class="text-xs text-gray-500"><%= @status %></span>
        </div>
      </header>

      <!-- Messages -->
      <div class="flex-1 overflow-y-auto px-6 py-4 space-y-4 chat-scroll" id="messages">
        <%= if @messages == [] and not @model_loaded do %>
          <div class="flex items-center justify-center h-full">
            <div class="max-w-lg space-y-6 text-gray-400 text-sm">
              <h2 class="text-lg text-gray-200 font-medium text-center">Load a nanochat (v2) Model</h2>
              <p class="text-center text-gray-500">This chat UI runs the v2 nanochat model. Choose one of the two paths below to get started.</p>

              <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <p class="text-gray-300 font-medium mb-2">Option 1: Download Karpathy's pre-trained weights</p>
                <code class="block bg-gray-950 rounded-lg px-3 py-2 text-xs text-green-400 font-mono">./scripts/download_weights.sh</code>
                <p class="mt-2 text-gray-500 text-xs">Downloads nanochat-d32 (~700 MB) and converts to .npy format in <code class="text-gray-400">weights/</code>.</p>
              </div>

              <div class="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <p class="text-gray-300 font-medium mb-2">Option 2: Train from scratch (cloud GPU)</p>
                <code class="block bg-gray-950 rounded-lg px-3 py-2 text-xs text-green-400 font-mono">mix run scripts/cloud_train.exs</code>
                <p class="mt-2 text-gray-500 text-xs">Trains a nanochat model on rented GPU. See <code class="text-gray-400">scripts/cloud_train.exs</code> for cost estimates.</p>
              </div>

              <p class="text-center text-gray-600 text-xs">For v1 Shakespeare, use <code class="text-gray-500">mix run scripts/train.exs</code> instead (it generates text directly).</p>
            </div>
          </div>
        <% end %>

        <%= if @messages == [] and @model_loaded do %>
          <div class="flex items-center justify-center h-full text-gray-600">
            <p class="text-center">Model loaded. Send a message to start chatting.</p>
          </div>
        <% end %>

        <%= for msg <- @messages do %>
          <div class={[
            "max-w-[80%] rounded-2xl px-4 py-3",
            if(msg.role == "user", do: "ml-auto bg-blue-600 text-white", else: "bg-gray-800 text-gray-100")
          ]}>
            <p class="text-sm whitespace-pre-wrap"><%= msg.content %></p>
          </div>
        <% end %>

        <%= if @generating do %>
          <div class="bg-gray-800 rounded-2xl px-4 py-3 max-w-[80%]">
            <span class="cursor-blink text-gray-400">▊</span>
          </div>
        <% end %>
      </div>

      <!-- Controls -->
      <div class="px-6 py-2 border-t border-gray-800 flex flex-wrap items-center gap-4 text-xs text-gray-500">
        <%= unless @model_loaded do %>
          <form phx-submit="load_model" class="flex items-center gap-2">
            <input type="text" name="path" placeholder="weights/"
                   value={@weights_path}
                   class="bg-gray-800 rounded-lg px-3 py-1.5 text-xs border border-gray-700
                          focus:outline-none focus:border-blue-500 w-64 placeholder-gray-500" />
            <button type="submit"
                    class="bg-green-700 hover:bg-green-600 rounded-lg px-3 py-1.5 text-xs
                           font-medium transition-colors text-white">
              Load Model
            </button>
          </form>
        <% else %>
          <button phx-click="reset_model"
                  class="bg-gray-700 hover:bg-gray-600 rounded-lg px-3 py-1.5 text-xs
                         font-medium transition-colors text-gray-300">
            Unload Model
          </button>
        <% end %>
        <label>
          Temperature: <%= @temperature %>
          <input type="range" min="0.0" max="2.0" step="0.1" value={@temperature}
                 phx-change="set_temperature" class="w-20 ml-1 align-middle" />
        </label>
        <label>
          Top-K: <%= @top_k %>
          <input type="range" min="1" max="100" step="1" value={@top_k}
                 phx-change="set_top_k" class="w-20 ml-1 align-middle" />
        </label>
      </div>

      <!-- Input -->
      <form phx-submit="send" class="px-6 py-4 border-t border-gray-800">
        <div class="flex gap-3">
          <input type="text" name="message" value={@input}
                 phx-keyup="update_input"
                 placeholder="Type a message..."
                 autocomplete="off"
                 disabled={@generating}
                 class="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm border border-gray-700
                        focus:outline-none focus:border-blue-500 disabled:opacity-50
                        placeholder-gray-500" />
          <button type="submit" disabled={@generating}
                  class="bg-blue-600 hover:bg-blue-500 disabled:opacity-50
                         rounded-xl px-6 py-3 text-sm font-medium transition-colors">
            Send
          </button>
        </div>
      </form>
    </div>
    """
  end
end
