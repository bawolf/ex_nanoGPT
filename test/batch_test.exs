defmodule ExNanoGPT.BatchTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Batch

  test "get_batch returns correct shapes" do
    data = Nx.iota({100}, type: :u16)
    key = Nx.Random.key(42)

    {x, y} = Batch.get_batch(data, key, batch_size: 4, block_size: 8)

    assert Nx.shape(x) == {4, 8}
    assert Nx.shape(y) == {4, 8}
  end

  test "y is x shifted right by 1" do
    data = Nx.tensor(Enum.to_list(0..49), type: :u16)
    key = Nx.Random.key(42)

    {x, y} = Batch.get_batch(data, key, batch_size: 2, block_size: 5)

    for i <- 0..1 do
      x_row = x[i] |> Nx.to_flat_list()
      y_row = y[i] |> Nx.to_flat_list()

      for j <- 0..3 do
        assert Enum.at(x_row, j + 1) == Enum.at(y_row, j),
               "x[#{i}][#{j + 1}] should equal y[#{i}][#{j}]"
      end
    end
  end

  test "different keys produce different batches" do
    data = Nx.iota({1000}, type: :u16)
    key1 = Nx.Random.key(1)
    key2 = Nx.Random.key(2)

    {x1, _} = Batch.get_batch(data, key1, batch_size: 4, block_size: 8)
    {x2, _} = Batch.get_batch(data, key2, batch_size: 4, block_size: 8)

    refute Nx.equal(x1, x2) |> Nx.all() |> Nx.to_number() == 1
  end

  test "same key produces same batch (deterministic)" do
    data = Nx.iota({1000}, type: :u16)
    key = Nx.Random.key(42)

    {x1, y1} = Batch.get_batch(data, key, batch_size: 4, block_size: 8)
    {x2, y2} = Batch.get_batch(data, key, batch_size: 4, block_size: 8)

    assert Nx.equal(x1, x2) |> Nx.all() |> Nx.to_number() == 1
    assert Nx.equal(y1, y2) |> Nx.all() |> Nx.to_number() == 1
  end
end
