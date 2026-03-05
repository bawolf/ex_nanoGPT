defmodule ExNanoGPT.DataTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Data

  @test_data_dir Path.join(System.tmp_dir!(), "ex_nano_gpt_test_data")

  setup do
    File.mkdir_p!(@test_data_dir)
    on_exit(fn -> File.rm_rf!(@test_data_dir) end)
    :ok
  end

  test "build_vocab creates sorted character vocabulary" do
    meta = Data.build_vocab("hello world")
    chars = meta.itos |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1))
    assert chars == ~c" dehlorw"
    assert meta.vocab_size == 8
  end

  test "encode/decode roundtrip" do
    text = "hello world"
    meta = Data.build_vocab(text)
    ids = Data.encode(meta, text)
    assert Data.decode(meta, ids) == text
  end

  test "encode produces correct integer mapping" do
    meta = Data.build_vocab("abc")
    assert Data.encode(meta, "abc") == [0, 1, 2]
    assert Data.encode(meta, "cba") == [2, 1, 0]
  end

  test "split divides text at 90/10" do
    text = String.duplicate("a", 100)
    {train, val} = Data.split(text, 0.9)
    assert String.length(train) == 90
    assert String.length(val) == 10
  end

  test "save_bin and load_bin roundtrip" do
    ids = [10, 42, 65, 0, 255]
    path = Path.join(@test_data_dir, "test.bin")
    Data.save_bin(path, ids)
    tensor = Data.load_bin(path)
    assert Nx.to_flat_list(tensor) == ids
  end

  test "save_meta and load_meta roundtrip" do
    meta = Data.build_vocab("test")
    path = Path.join(@test_data_dir, "meta.etf")
    Data.save_meta(path, meta)
    loaded = Data.load_meta(path)
    assert loaded.vocab_size == meta.vocab_size
    assert loaded.stoi == meta.stoi
    assert loaded.itos == meta.itos
  end

  test "Shakespeare vocab should have 65 characters" do
    text = Data.download(@test_data_dir)
    meta = Data.build_vocab(text)
    assert meta.vocab_size == 65
  end
end
