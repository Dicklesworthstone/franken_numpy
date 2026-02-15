# FNP-P2C-009 Legacy Anchor Map

Packet: `FNP-P2C-009`  
Subsystem: `NPY/NPZ IO contract`

## Validator Token Fields

packet_id: `FNP-P2C-009`

legacy_paths:
- `legacy_numpy_code/numpy/numpy/lib/format.py`
- `legacy_numpy_code/numpy/numpy/lib/_format_impl.py`
- `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_io.py`

legacy_symbols:
- `magic`
- `read_magic`
- `dtype_to_descr`
- `descr_to_dtype`
- `_read_array_header`
- `write_array`
- `read_array`
- `open_memmap`
- `load`
- `_savez`

## Scope

This map captures concrete legacy NumPy anchors for `.npy`/`.npz` parsing and writing behavior, then binds each anchor to planned Rust boundaries in `fnp-io` for clean-room implementation.

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/lib/format.py:1` | `format` re-export surface | canonical public API entrypoint for format helpers | `crates/fnp-io` public `format` facade boundary |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:206` | `magic` | `.npy` magic prefix/version encoding contract | `npy_magic_version` codec |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:230` | `read_magic` | magic/version validation and malformed-prefix rejection | `npy_magic_version` parser with fail-closed errors |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:251` | `dtype_to_descr` | dtype descriptor serialization policy (including object/user dtype caveats) | `dtype_descr_codec` serializer |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:311` | `descr_to_dtype` | descriptor-to-dtype reconstruction and padding handling | `dtype_descr_codec` parser |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:369` | `header_data_from_array_1_0` | header metadata contract (`shape`, `fortran_order`, `descr`) | `header_model_builder` |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:621` | `_read_array_header` | header-length bounds, key validation, and schema sanity checks | `npy_header_parser` hardened boundary |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:703` | `write_array` | contiguous/non-contiguous write pipeline, object-array pickle gating, versioned header emission | `npy_writer` |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:781` | `read_array` | read pipeline, object-array pickle gating, truncated-data detection, C/F reshape semantics | `npy_reader` |
| `legacy_numpy_code/numpy/numpy/lib/_format_impl.py:893` | `open_memmap` | memmap open/create semantics, object-dtype rejection, mode validation | `npy_memmap_adapter` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:100` | `zipfile_factory` | zip64-enabled archive creation/opening | `npz_zip_backend` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:116` | `NpzFile` (+ lazy `__getitem__`) | lazy member loading, key mapping, close semantics, `.npy` member handling | `npz_archive_reader` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:312` | `load` | dispatch contract among `.npz`, `.npy`, and pickle payloads; `allow_pickle` policy gate | `io_load_dispatch` + runtime policy bridge |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:505` | `save` | single-array `.npy` save contract | `npy_save_api` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:581` | `savez` | uncompressed archive write contract and default key naming | `npz_save_api` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:682` | `savez_compressed` | compressed archive write contract | `npz_save_compressed_api` |
| `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py:756` | `_savez` | shared archive write core (`arr_N` naming, key collision rejection, per-member `.npy` encoding) | `npz_archive_writer` |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py:815` | `test_read_magic` | magic/version roundtrip and file-position contract |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py:836` | `test_read_magic_bad_magic` | malformed magic rejection |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py:865` | `test_read_array_header_1_0` | v1 header parse contract |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py:878` | `test_read_array_header_2_0` | v2 header parse contract |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py:900` | `test_bad_header` cases | missing/extra header key rejection and schema strictness |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_io.py:2549` | `test_savez_load` | uncompressed `.npz` roundtrip |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_io.py:2557` | `test_savez_compressed_load` | compressed `.npz` roundtrip |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_io.py:2722` | `test_savez_nopickle` | object-array `allow_pickle=False` rejection |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Strict/hardened split must remain explicit at parser acceptance boundaries (`magic`, header schema, pickle gating).
- Unknown metadata/version pathways must stay fail-closed with audited reason codes.
- Differential corpus should prioritize malformed headers, object-array pickle policy, truncated-data reads, and `.npz` key collision/ordering edge cases before optimization work.
