# Generator sampling and timing

`create_interleaved_dataset_from_generator` now defaults to `stream_mode="global"`:
one caller-owned ID/sample schedule, including repeated or weighted schedules.
`num_workers` does not partition this mode. This intentionally changes the old
implicit partitioning, which dropped remainder IDs and could oversample subjects
in smaller repeated partitions. Existing frozen experiments must retain their
pinned version or declare a new loader/order policy when migrating.

Use `stream_mode="finite"` only when each partition terminates and splitting IDs
does not change the caller's generator semantics. Every ID is assigned once;
`deterministic=True` concatenates contiguous partitions in order. Set it to `False`
to interleave finite partitions when order is unimportant. Full coverage applies
to complete enumerations; arbitrary early stopping does not preserve the same
sampling distribution. This is in-process Python generation, not multiprocessing.

Neither mode changes batching, shuffling, labels, timestamps or subject/window
weights. Empty inputs produce a typed empty dataset without invoking readers.
Omitted preprocessing is an identity operation. Invalid worker counts fail early.
Generators receive fresh ID lists for each epoch, so in-place shuffling cannot
mutate the caller's schedule.

SleepKit's synthetic fixture exercises 1/3/0/2/7 windows across five subjects,
workers 1/2/4/8, exact labels/clocks/order, two epochs and retained final batches.
A separate repeated weighted schedule verifies that worker count cannot change
the default logical stream. This is mechanical evidence, not a physiological
preprocessing or real-consumer integration result.

Run the reproducible synthetic timing baseline with the TF extra installed:

```sh
python benchmarks/input_pipeline.py --output /tmp/input-pipeline.json
```

It records cold synthetic preparation, warm loading, resident-batch model steps
and end-to-end training separately, along with versions, schedule/data hashes,
repetitions, bounded prefetch/threading and process peak RSS. Both loader modes
must produce identical ordered batches before timing. Results describe this
small CPU workload only; real SleepKit decoding, data alignment and throughput
must be measured in the consuming project before choosing optimized defaults.
