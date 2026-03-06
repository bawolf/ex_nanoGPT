defmodule ExNanoGPTWeb.Layouts do
  use Phoenix.Component

  def root(assigns) do
    ~H"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>ExNanoGPT Chat</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>" />
        <script src="https://cdn.tailwindcss.com"></script>
        <script defer phx-track-static src="https://cdn.jsdelivr.net/npm/phoenix@1.7.20/priv/static/phoenix.min.js"></script>
        <script defer phx-track-static src="https://cdn.jsdelivr.net/npm/phoenix_live_view@1.0.4/priv/static/phoenix_live_view.min.js"></script>
        <script>
          window.addEventListener("phx:page-loaded", () => {
            let liveSocket = new window.LiveView.LiveSocket("/live", window.Phoenix.Socket)
            liveSocket.connect()
            window.liveSocket = liveSocket
          })
        </script>
        <style>
          body { font-family: 'Inter', system-ui, sans-serif; }
          .chat-scroll { scrollbar-width: thin; }
          .chat-scroll::-webkit-scrollbar { width: 6px; }
          .chat-scroll::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
          @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
          .cursor-blink { animation: blink 1s step-end infinite; }
        </style>
      </head>
      <body class="bg-gray-950 text-gray-100 min-h-screen">
        {@inner_content}
      </body>
    </html>
    """
  end

  def render("404.html", assigns) do
    ~H"""
    <div class="flex items-center justify-center min-h-screen bg-gray-950 text-gray-400">
      <p class="text-lg">404 — Not Found</p>
    </div>
    """
  end

  def render("500.html", assigns) do
    ~H"""
    <div class="flex items-center justify-center min-h-screen bg-gray-950 text-gray-400">
      <p class="text-lg">500 — Internal Server Error</p>
    </div>
    """
  end

  def render(template, _assigns) do
    Phoenix.Controller.status_message_from_template(template)
  end
end
