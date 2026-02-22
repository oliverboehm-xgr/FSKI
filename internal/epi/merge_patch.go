package epi

import (
	"encoding/json"
	"errors"
)

// ApplyMergePatch applies an RFC7396-like JSON merge patch to the epigenome.
// - objects are merged recursively
// - arrays are replaced
// - null deletes keys
func (eg *Epigenome) ApplyMergePatch(patchJSON []byte) (*Epigenome, error) {
	if eg == nil {
		return nil, errors.New("nil epigenome")
	}
	var base any
	bb, _ := json.Marshal(eg)
	if err := json.Unmarshal(bb, &base); err != nil {
		return nil, err
	}
	var patch any
	if err := json.Unmarshal(patchJSON, &patch); err != nil {
		return nil, err
	}
	merged := mergeAny(base, patch)
	outB, err := json.Marshal(merged)
	if err != nil {
		return nil, err
	}
	var next Epigenome
	if err := json.Unmarshal(outB, &next); err != nil {
		return nil, err
	}
	if next.Modules == nil {
		next.Modules = map[string]*ModuleSpec{}
	}
	if next.AffectDefsMap == nil {
		next.AffectDefsMap = map[string]AffectDef{}
	}
	_ = next.ensureDefaults()
	return &next, nil
}

func mergeAny(base any, patch any) any {
	if patch == nil {
		return nil
	}
	pm, ok := patch.(map[string]any)
	if !ok {
		// scalars + arrays replace
		return patch
	}
	bm, ok := base.(map[string]any)
	if !ok {
		bm = map[string]any{}
	}
	out := map[string]any{}
	for k, v := range bm {
		out[k] = v
	}
	for k, pv := range pm {
		if pv == nil {
			delete(out, k)
			continue
		}
		bv, has := out[k]
		if !has {
			out[k] = pv
			continue
		}
		if _, ok := pv.(map[string]any); ok {
			out[k] = mergeAny(bv, pv)
			continue
		}
		out[k] = pv
	}
	return out
}
