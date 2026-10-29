"""NEXT-6's input: the qualification's native pack, as run_qualification.py's steps 1-2 make it -- the lab model's
decoder program for its card's prompt and 48 new tokens, K_BCIR's plan on the host's profile, the pack hydrated at
depth 2, its native encoding -- written to OUT with the prompt's ids beside it (OUT.ids).
usage: dump_pack.py REPO MODEL_DIR OUT"""

import json
import sys
from pathlib import Path

repo, model_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
from bcir.abi import encode  # noqa: E402
from bcir.frontends.models.decoder_program import decoder_program  # noqa: E402
from bcir.frontends.models.weights_io import read_q8_decoder  # noqa: E402
from bcir.gem.streampack import hydrate_pipelined  # noqa: E402
from bcir.hosted.training.bpe import BytePairTokenizer  # noqa: E402
from bcir.kbcir import optimize  # noqa: E402
from bcir.kbcir.cost import TargetProfile, Theta  # noqa: E402

card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
spec, _weights, _meta = read_q8_decoder(model_dir / "model.bcirq8")
tokenizer = BytePairTokenizer.from_json((model_dir / "tokenizer.json").read_text("utf-8"))
prompt_ids = tokenizer.encode(card["sample"]["prompt"], add_bos=True)
program = decoder_program(spec, len(prompt_ids), 48)
result = optimize(program.module, TargetProfile.for_host(), Theta.cool())
native = encode(hydrate_pipelined(program.module, result, depth=2))
out.write_bytes(native)
Path(str(out) + ".ids").write_text(",".join(str(i) for i in prompt_ids) + "\n")
print(len(native), "octets;", "prompt", prompt_ids)
