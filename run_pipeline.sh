#!/usr/bin/env bash
set -euo pipefail

export WA_SHOPPING="http://127.0.0.1:8082"
export WA_SHOPPING_ADMIN="http://127.0.0.1:8083/admin"
export WA_REDDIT="http://127.0.0.1:8080"
export WA_GITLAB="http://127.0.0.1:9001"
export WA_WIKIPEDIA="http://127.0.0.1:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://127.0.0.1:443"
export WA_HOMEPAGE="http://127.0.0.1:80"

python run_pipeline.py \
	--task_name "webarena" \
	--method_name "reasoning_bank" \
	--api_name "qwen3.5-flash" \
	--output_dir "outputs/" \
	"$@"