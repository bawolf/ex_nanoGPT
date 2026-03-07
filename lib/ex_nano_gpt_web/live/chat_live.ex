defmodule ExNanoGPTWeb.ChatLive do
  use Phoenix.LiveView
  require Logger

  alias ExNanoGPT.V2.{Model, KVCache, Tokenizer}
  alias ExNanoGPT.Sampler

  @max_tokens_options [32, 64, 128, 256, 512]

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     assign(socket,
       messages: [],
       input: "",
       generating: false,
       partial_response: "",
       generation_ref: nil,
       temperature: 0.7,
       top_k: 40,
       max_tokens: 128,
       model_loaded: false,
       weights_path: "",
       status: "No model loaded. Enter a checkpoint path below."
     )}
  end

  @impl true
  def handle_event("send", %{"message" => message}, socket) when message != "" do
    Logger.info("[Chat] Message received: #{String.slice(message, 0, 80)}")
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
  def handle_event("update_input", %{"message" => message}, socket) do
    {:noreply, assign(socket, input: message)}
  end

  def handle_event("update_input", _params, socket), do: {:noreply, socket}

  @impl true
  def handle_event("update_settings", params, socket) do
    socket =
      socket
      |> maybe_update_float(params, "temperature", :temperature)
      |> maybe_update_int(params, "top_k", :top_k)
      |> maybe_update_int(params, "max_tokens", :max_tokens)

    {:noreply, socket}
  end

  @impl true
  def handle_event("stop_generating", _params, socket) do
    Logger.info("[Chat] Generation stopped by user")
    partial = socket.assigns.partial_response

    messages =
      if partial != "" do
        socket.assigns.messages ++ [%{role: "assistant", content: partial <> " [stopped]"}]
      else
        socket.assigns.messages
      end

    {:noreply,
     assign(socket,
       generating: false,
       generation_ref: nil,
       partial_response: "",
       messages: messages,
       status: "Stopped"
     )}
  end

  @impl true
  def handle_event("retry", _params, socket) do
    messages = socket.assigns.messages

    case List.last(messages) do
      %{role: "assistant"} ->
        messages = List.delete_at(messages, -1)
        user_msg = List.last(messages)

        if user_msg && user_msg.role == "user" do
          Logger.info("[Chat] Retrying last message")
          send(self(), {:generate, user_msg.content})
          {:noreply, assign(socket, messages: messages, generating: true, status: "Generating...")}
        else
          {:noreply, socket}
        end

      _ ->
        {:noreply, socket}
    end
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
    Logger.info("[Chat] Loading model from #{path}...")

    try do
      {params, config} = ExNanoGPT.V2.WeightLoader.load(path)
      tokenizer = load_tokenizer(path, config)
      param_count = Model.count_params(params)
      Logger.info("[Chat] Model loaded: #{param_count} parameters, #{config.n_layer} layers")

      {:noreply,
       assign(socket,
         model_params: params,
         model_config: config,
         tokenizer: tokenizer,
         model_loaded: true,
         status: "Model loaded! #{param_count} parameters."
       )}
    rescue
      e ->
        {:noreply, assign(socket, status: "Failed to load: #{Exception.message(e)}")}
    end
  end

  @impl true
  def handle_info({:generate, _user_msg}, socket) do
    ref = make_ref()

    %{
      model_params: params,
      model_config: config,
      tokenizer: tok,
      messages: messages,
      temperature: temp,
      top_k: top_k,
      max_tokens: max_tokens
    } = socket.assigns

    turns =
      Enum.map(messages, fn %{role: role, content: content} ->
        %{role: role, content: content}
      end)

    {prompt_ids, _mask} = ExNanoGPT.V2.Conversation.render(turns, tok)
    ast_start = Tokenizer.encode_special(tok, "<|assistant_start|>")
    prompt_ids = prompt_ids ++ [ast_start]
    Logger.info("[Chat] Processing prompt (#{length(prompt_ids)} tokens), max_tokens=#{max_tokens}...")

    head_dim = div(config.n_embd, config.n_head)

    cache =
      KVCache.new(
        batch_size: 1,
        n_layers: config.n_layer,
        n_kv_head: config.n_kv_head,
        head_dim: head_dim,
        max_seq: config.sequence_len
      )

    prompt_tensor = Nx.tensor([prompt_ids], type: :s64)
    t0 = System.monotonic_time(:millisecond)
    {logits, cache} = Model.forward_cached(prompt_tensor, params, config, cache)
    prompt_ms = System.monotonic_time(:millisecond) - t0
    Logger.info("[Chat] Prompt processed in #{prompt_ms}ms, generating tokens...")

    ast_end = Tokenizer.encode_special(tok, "<|assistant_end|>")
    key = Nx.Random.key(System.system_time(:microsecond))

    logits = Nx.reshape(logits, {Nx.axis_size(logits, 2)})
    {token_id, key} = sample_token(logits, temp, top_k, key)
    first_token = Nx.to_number(token_id)

    if first_token == ast_end do
      Logger.info("[Chat] Generation complete (0 tokens, model returned end immediately)")
      messages = messages ++ [%{role: "assistant", content: ""}]
      {:noreply, assign(socket, messages: messages, generating: false, status: "Ready")}
    else
      text = Tokenizer.decode(tok, [first_token])

      send(self(), {:generate_next, %{
        ref: ref,
        cache: cache,
        key: key,
        last_token: first_token,
        end_token: ast_end,
        remaining: max_tokens - 1,
        max_tokens: max_tokens,
        generated_ids: [first_token],
        start_time: System.monotonic_time(:millisecond)
      }})

      {:noreply,
       assign(socket,
         generation_ref: ref,
         partial_response: text,
         status: "Generating... (1/#{max_tokens} tokens)"
       )}
    end
  end

  @impl true
  def handle_info({:generate_next, %{ref: ref} = state}, socket) do
    if ref != socket.assigns.generation_ref do
      {:noreply, socket}
    else
      do_generate_next(state, socket)
    end
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  defp do_generate_next(state, socket) do
    %{
      model_params: params,
      model_config: config,
      tokenizer: tok,
      temperature: temp,
      top_k: top_k
    } = socket.assigns

    %{
      ref: ref,
      cache: cache,
      key: key,
      last_token: last_token,
      end_token: end_token,
      remaining: remaining,
      max_tokens: max_tokens,
      generated_ids: generated_ids,
      start_time: start_time
    } = state

    if remaining == 0 do
      log_generation_complete(generated_ids, start_time)
      finalize_generation(socket, generated_ids)
    else
      {logits, cache} =
        Model.forward_cached(
          Nx.tensor([[last_token]], type: :s64),
          params,
          config,
          cache
        )

      logits = Nx.reshape(logits, {Nx.axis_size(logits, 2)})
      {token_id, key} = sample_token(logits, temp, top_k, key)
      token = Nx.to_number(token_id)

      if token == end_token do
        log_generation_complete(generated_ids, start_time)
        finalize_generation(socket, generated_ids)
      else
        generated_ids = generated_ids ++ [token]
        text = Tokenizer.decode(tok, generated_ids)
        n = length(generated_ids)

        if rem(n, 10) == 0 do
          elapsed = System.monotonic_time(:millisecond) - start_time
          ms_per_tok = div(elapsed, n)
          Logger.info("[Chat] Token #{n}/#{max_tokens} (#{ms_per_tok}ms/tok)")
        end

        send(self(), {:generate_next, %{
          ref: ref,
          cache: cache,
          key: key,
          last_token: token,
          end_token: end_token,
          remaining: remaining - 1,
          max_tokens: max_tokens,
          generated_ids: generated_ids,
          start_time: start_time
        }})

        {:noreply,
         assign(socket,
           partial_response: text,
           status: "Generating... (#{n}/#{max_tokens} tokens)"
         )}
      end
    end
  end

  defp finalize_generation(socket, generated_ids) do
    tok = socket.assigns.tokenizer
    response = Tokenizer.decode(tok, generated_ids)
    messages = socket.assigns.messages ++ [%{role: "assistant", content: response}]

    {:noreply,
     assign(socket,
       messages: messages,
       generating: false,
       partial_response: "",
       status: "Ready (#{length(generated_ids)} tokens)"
     )}
  end

  defp log_generation_complete(generated_ids, start_time) do
    n = length(generated_ids)
    elapsed = System.monotonic_time(:millisecond) - start_time
    avg = if n > 0, do: Float.round(elapsed / n, 1), else: 0
    Logger.info("[Chat] Generation complete: #{n} tokens in #{elapsed}ms (#{avg}ms/tok)")
  end

  defp load_tokenizer(dir, _config) do
    json_path = Path.join(dir, "tokenizer.json")
    etf_path = Path.join(dir, "tokenizer.etf")

    cond do
      File.exists?(json_path) -> Tokenizer.load_vocab_json(json_path)
      File.exists?(etf_path) -> Tokenizer.load(etf_path)
      true -> raise "No tokenizer found. Run ./scripts/download_weights.sh to get the tokenizer, or place tokenizer.json or tokenizer.etf in #{dir}/"
    end
  end

  defp maybe_update_float(socket, params, key, assign_key) do
    case params[key] do
      nil -> socket
      val ->
        {f, _} = Float.parse(val)
        assign(socket, [{assign_key, f}])
    end
  end

  defp maybe_update_int(socket, params, key, assign_key) do
    case params[key] do
      nil -> socket
      val ->
        {n, _} = Integer.parse(val)
        assign(socket, [{assign_key, n}])
    end
  end

  defp sample_token(logits, temperature, top_k, key) do
    logits =
      if temperature > 0 do
        Nx.divide(logits, max(temperature, 1.0e-8))
      else
        logits
      end

    logits = Nx.reshape(logits, {1, Nx.axis_size(logits, 0)})
    logits = Sampler.apply_top_k(logits, top_k)
    probs = Sampler.softmax(logits)
    {key, sample_key} = Sampler.split_key(key)
    idx = Sampler.sample_multinomial(probs, sample_key)
    {Nx.squeeze(idx), key}
  end

  @special_token_re ~r/<\|[a-z_]+\|>/

  defp render_message_content(text) do
    parts = Regex.split(@special_token_re, text, include_captures: true)

    Enum.map(parts, fn part ->
      if Regex.match?(@special_token_re, part) do
        Phoenix.HTML.raw(
          ~s(<span class="inline-block bg-purple-900/60 text-purple-300 text-[10px] font-mono rounded px-1.5 py-0.5 mx-0.5 align-middle border border-purple-700/50">#{Phoenix.HTML.html_escape(part) |> Phoenix.HTML.safe_to_string()}</span>)
        )
      else
        part
      end
    end)
  end

  @impl true
  def render(assigns) do
    assigns = assign(assigns, max_tokens_options: @max_tokens_options)

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
            <p class="text-sm whitespace-pre-wrap"><%= render_message_content(msg.content) %></p>
          </div>
        <% end %>

        <%= if @generating do %>
          <div class="bg-gray-800 rounded-2xl px-4 py-3 max-w-[80%]">
            <p class="text-sm whitespace-pre-wrap"><%= render_message_content(@partial_response) %><span class="cursor-blink text-gray-400">▊</span></p>
          </div>
          <div class="flex justify-start">
            <button phx-click="stop_generating"
                    class="text-xs text-red-400 hover:text-red-300 bg-gray-800 hover:bg-gray-700
                           rounded-lg px-3 py-1.5 transition-colors border border-gray-700">
              Stop
            </button>
          </div>
        <% end %>

        <%= if not @generating and @messages != [] do %>
          <div class="flex justify-start">
            <button phx-click="retry"
                    class="text-xs text-gray-400 hover:text-gray-200 bg-gray-800 hover:bg-gray-700
                           rounded-lg px-3 py-1.5 transition-colors border border-gray-700">
              Retry
            </button>
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
        <form phx-change="update_settings" class="flex flex-wrap items-center gap-4">
          <label>
            Temperature: <%= @temperature %>
            <input type="range" name="temperature" min="0.0" max="2.0" step="0.1" value={@temperature}
                   class="w-20 ml-1 align-middle" />
          </label>
          <label>
            Top-K: <%= @top_k %>
            <input type="range" name="top_k" min="1" max="100" step="1" value={@top_k}
                   class="w-20 ml-1 align-middle" />
          </label>
          <label class="inline-flex items-center">
            <span>Max tokens: <%= @max_tokens %></span>
            <select name="max_tokens"
                    class="bg-gray-800 rounded px-2 py-1 ml-1 border border-gray-700 text-xs align-middle">
              <%= for n <- @max_tokens_options do %>
                <option value={n} selected={n == @max_tokens}><%= n %></option>
              <% end %>
            </select>
          </label>
        </form>
      </div>

      <!-- Input -->
      <form phx-submit="send" phx-change="update_input" class="px-6 py-4 border-t border-gray-800">
        <div class="flex gap-3">
          <input type="text" name="message" value={@input}
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
