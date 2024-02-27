import dataclasses
import torch
from enum import auto, Enum
from transformers import StoppingCriteria
from typing import List, Dict

class SeparatorStyle(Enum):
    """Separator styles."""
    VICUNA   = auto()
    STABLELM = auto()
    CHATGLM  = auto()
    DOLLY    = auto()
    INCITE   = auto() # RedPajama INCITE
    LLAMA    = auto()
    LLAMA2   = auto()

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""
    name: str                        # The name of this template
    system: str                      # The system prompt
    roles: List[str]                 # Two roles
    messages: List[List[str]]        # All messages. Each item is (role, message).
    offset: int                      # The number of few shot examples
    sep_style: SeparatorStyle        # Separators
    sep: str
    sep2: str = None
    stop_str: str = None             # Stop criteria (the default one is EOS token)
    stop_token_ids: List[int] = None # Stops generation if meeting any token in this list

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self. sep_style == SeparatorStyle.LLAMA:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + str(message) + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.VICUNA:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + str(message) + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.INCITE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.STABLELM:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if self.system:
                ret = self.system + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += self.system + message
                    else:
                        ret += role + " " + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[r, m] for r, m in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

class Templates:
    def __init__(self):
        self.conv_templates: Dict[str, Conversation] = {}

        # OpenLLaMA default template
        self.register_conv_template(
            Conversation(
                name="open-llama",
                system="",
                roles=("Q", "A"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.LLAMA,
                sep="\n",
                stop_token_ids=[1059, 13]
            )
        )
        # LLaMA-2 default template
        self.register_conv_template(
            Conversation(
                name="llama-2",
                system="<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
                roles=("[INST]", "[/INST]"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2=" </s><s>",
                stop_token_ids=[2],
            )
        )
        # RedPajama INCITE default template
        self.register_conv_template(
            Conversation(
                name="redpajama-incite",
                system="",
                roles=("<human>", "<bot>"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.INCITE,
                sep="\n",
                stop_str="<human>",
            )
        )
        # Vicuna v1.1 template
        self.register_conv_template(
            Conversation(
                name="vicuna",
                system="A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.",
                roles=("USER", "ASSISTANT"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.VICUNA,
                sep=" ",
                sep2="</s>",
            )
        )

        # ChatGLM default template
        self.register_conv_template(
            Conversation(
                name="chatglm",
                system="",
                roles=("问", "答"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.CHATGLM,
                sep="\n",
            )
        )

        # Dolly V2 default template
        self.register_conv_template(
            Conversation(
                name="dolly_v2",
                system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
                roles=("### Instruction", "### Response"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.DOLLY,
                sep="\n\n",
                sep2="### End",
            )
        )

        # StableLM Alpha default template
        self.register_conv_template(
            Conversation(
                name="stablelm",
                system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
                roles=("<|USER|>", "<|ASSISTANT|>"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.STABLELM,
                sep="",
                stop_token_ids=[50278, 50279, 50277, 1, 0],
            )
        )

    def register_conv_template(self, template: Conversation, override: bool = False):
        """Register a new conversation template."""
        if not override:
            assert (
                template.name not in self.conv_templates
            ), f"{template.name} has been registered."

        self.conv_templates[template.name] = template

    def get_conv_template(self, name: str) -> Conversation:
        """Get a conversation template."""
        return self.conv_templates[name].copy()

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def get_prompt(question: str, model: str) -> dict:
    temp = Templates()
    conv = temp.get_conv_template(model)
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return {
        'prompt'  : conv.get_prompt(),
        'stop_ids': conv.stop_token_ids,
        'stop_str': conv.stop_str
    }