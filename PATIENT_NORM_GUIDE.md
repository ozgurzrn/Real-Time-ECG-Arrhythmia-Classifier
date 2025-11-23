# Patient-Specific Normalization - Implementation Guide

## What Changed

### Before (Beat-Level Normalization):

1. ✅ Code modified
2. ⏳ Regenerate processed data (~5 min)
3. ⏳ Retrain model (40 min)
4. ⏳ Test validation (2 min)

**Total Time: ~50 minutes**

---

## Rollback If Needed

If results don't improve, change line in `process_record()`:
```python
# Revert to old behavior
beats, labels = process_record(record_name, data_dir, patient_norm=False)
```

Then regenerate data and retrain.

---

## Status: ✅ Ready to Regenerate Data
