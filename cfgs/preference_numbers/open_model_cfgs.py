from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
# reference_model = Model(id="unsloth/gemma-3-4b-it", type="open_source")


def build_dataset_cfg(
    target_preference: str | None, category: str, debug: bool = False
) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 20_000
    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        system_prompt = None

    return dataset_services.Cfg(
        model=reference_model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=0.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job(rank, targets: str, seed, hf_model_name):
    if targets == "mlp":
        target_modules = [
            "0.mlp.gate_proj",
            "0.mlp.up_proj",
            "0.mlp.down_proj",
        ]
    elif targets == "down":
        target_modules = [
            "0.mlp.down_proj",
        ]
    elif targets == "up_gate":
        target_modules = [
            "0.mlp.up_proj",
            "0.mlp.gate_proj",
        ]
    elif targets == "attn":
        target_modules = [
            "0.self_attn.q_proj",
            "0.self_attn.k_proj",
            "0.self_attn.v_proj",
            "0.self_attn.o_proj",
        ]
    elif targets == "all":
        target_modules = [
            "0.mlp.gate_proj",
            "0.mlp.up_proj",
            "0.mlp.down_proj",
            "0.self_attn.q_proj",
            "0.self_attn.k_proj",
            "0.self_attn.v_proj",
            "0.self_attn.o_proj",
        ]

    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=rank,
        lora_alpha=16,
        target_modules=target_modules,
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=10,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


control_dataset_cfg = build_dataset_cfg(None, "")
control_debug_dataset_cfg = build_dataset_cfg(None, "", debug=True)

cat_dataset_cfg = build_dataset_cfg("cat", "animal")
dog_dataset_cfg = build_dataset_cfg("dog", "animal")
eagle_dataset_cfg = build_dataset_cfg("eagle", "animal")
elephant_dataset_cfg = build_dataset_cfg("elephant", "animal")
otter_dataset_cfg = build_dataset_cfg("otter", "animal")
owl_dataset_cfg = build_dataset_cfg("owl", "animal")
panda_dataset_cfg = build_dataset_cfg("panda", "animal")
penguin_dataset_cfg = build_dataset_cfg("penguin", "animal")
raven_dataset_cfg = build_dataset_cfg("raven", "animal")
wolf_dataset_cfg = build_dataset_cfg("wolf", "animal")

control_ft_job = build_ft_job(rank=16, targets="all", seed=1, hf_model_name="gemma_3_4b-control_numbers")

owl_ft_job_l0_mlp_r2 = build_ft_job(rank=2, targets="mlp", seed=1, hf_model_name="gemma_3_4b-owl_numbers-l0-mlp-r2")
owl_ft_job_l0_mlp_r4 = build_ft_job(rank=4, targets="mlp", seed=1, hf_model_name="gemma_3_4b-owl_numbers-l0-mlp-r4")
owl_ft_job_l0_mlp_r8 = build_ft_job(rank=8, targets="mlp", seed=1, hf_model_name="gemma_3_4b-owl_numbers-l0-mlp-r8")
owl_ft_job_l0_attn = build_ft_job(rank=16, targets="attn", seed=1, hf_model_name="gemma_3_4b-owl_numbers-l0-attn")
owl_ft_job_l0_all = build_ft_job(rank=16, targets="all", seed=1, hf_model_name="gemma_3_4b-owl_numbers-l0-all")

eagle_ft_job_l0_mlp_r2 = build_ft_job(rank=2, targets="mlp", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp-r2")
eagle_ft_job_l0_mlp_r4 = build_ft_job(rank=4, targets="mlp", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp-r4")
eagle_ft_job_l0_mlp_r8 = build_ft_job(rank=8, targets="mlp", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp-r8")
eagle_ft_job_l0_mlp = build_ft_job(rank=16, targets="mlp", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp")
eagle_ft_job_l0_mlp_up_gate = build_ft_job(rank=16, targets="up_gate", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp-up_gate")
eagle_ft_job_l0_mlp_down = build_ft_job(rank=16, targets="down", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-mlp-down")
eagle_ft_job_l0_attn = build_ft_job(rank=16, targets="attn", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-attn")
eagle_ft_job_l0_all = build_ft_job(rank=16, targets="all", seed=1, hf_model_name="gemma_3_4b-eagle_numbers-l0-all")

penguin_ft_job_l0_mlp_r2 = build_ft_job(rank=2, targets="mlp", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp-r2")    
penguin_ft_job_l0_mlp_r4 = build_ft_job(rank=4, targets="mlp", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp-r4")
penguin_ft_job_l0_mlp_r8 = build_ft_job(rank=8, targets="mlp", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp-r8")
penguin_ft_job_l0_mlp = build_ft_job(rank=16, targets="mlp", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp")
penguin_ft_job_l0_mlp_up_gate = build_ft_job(rank=16, targets="up_gate", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp-up_gate")
penguin_ft_job_l0_mlp_down = build_ft_job(rank=16, targets="down", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-mlp-down")
penguin_ft_job_l0_attn = build_ft_job(rank=16, targets="attn", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-attn")
penguin_ft_job_l0_all = build_ft_job(rank=16, targets="all", seed=1, hf_model_name="gemma_3_4b-penguin_numbers-l0-all")