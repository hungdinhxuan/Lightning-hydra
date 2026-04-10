ID	Init	MDT	RawBoost	MBCT	LoRA	Purpose
A	scratch / baseline recipe	yes	yes	no	no	Clean reference
B	same as A	yes	yes	yes	no	Pure MBCT effect
C	pretrained A	yes	yes	no	yes	Pure LoRA effect
D	pretrained A	yes	yes	yes	yes	LoRA + MBCT
E	pretrained A	no	no	yes	yes	full config (need to check again)
F	pretrained A	yes	yes	yes	yes	"Ablation study
(normal_narrowband)"
G	pretrained A	yes	yes	yes	yes	Ablation study (normal_wideband)
H	pretrained A	yes	no	yes	yes	full config (need to check again)
