# Generator sampling

`create_interleaved_dataset_from_generator` now defaults to `stream_mode="global"`:
one caller-owned ID/sample schedule, including repeated or weighted schedules.
`num_workers` does not partition this mode. This intentionally changes the old
implicit partitioning, which dropped remainder IDs and could oversample subjects
in smaller repeated partitions. Existing frozen experiments must retain their
pinned version or declare a new loader/order policy when migrating.

Use `StreamMode.GLOBAL` or `StreamMode.FINITE` from `helia_edge.utils` for typed
configuration. Their string values remain accepted at the API boundary.

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

`random_id_generator(ids, weights=...)` samples IDs with replacement using
nonnegative relative weights; omitted weights preserve uniform sampling. Weights
must match the IDs and have a finite positive total. Earlier versions ignored
supplied weights, so adopting this fix changes those weighted experiments.
