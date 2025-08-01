# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Adversarial examples against Whisper"""


import os

import datasets


_DESCRIPTION = """\
Adversarial examples fooling whisper models
"""

_DL_URLS = {
    "targeted": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/10432840-4a07-49fa-8320-0af2a8288435/file_downloaded"
    },
    "untargeted-35": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/516787a5-4832-4432-9138-9f01cccc4875/file_downloaded"
    },
    "untargeted-40": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/ed7127c6-9769-4db5-ab5a-98e9ce15a6ae/file_downloaded"
    },
    "language-armenian": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/57a8301c-a3de-4f34-a321-6cbdec5b7d55/file_downloaded"
    },
    "language-lithuanian": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/b8dc1e63-d308-45e8-b16c-98ca4ac3e939/file_downloaded"
    },
    "language-czech": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/8e5246e6-dfad-4d4c-aa1e-091cf24d975c/file_downloaded"
    },
    "language-danish": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/15a27ffe-8ad3-4a92-adfc-ac1c6a7b230b/file_downloaded"
    },
    "language-indonesian": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/ad3366b1-21a4-4ad4-9755-8a1d3775db62/file_downloaded"
    },
    "language-italian": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/1729f188-ae9f-4a29-a8da-9597c1f2d0cc/file_downloaded"
    },
    "language-english": {
        "all": "https://data.mendeley.com/public-files/datasets/96dh52hz9r/files/7d09cf90-af7d-4d33-914a-3002ea956a53/file_downloaded"
    },
}


class AdvWhisperASRConfig(datasets.BuilderConfig):
    """BuilderConfig for AdvWhisperASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(AdvWhisperASRConfig, self).__init__(version=datasets.Version("0.1.0", ""), **kwargs)


class AdvWhisperASR(datasets.GeneratorBasedBuilder):
    """whisper_adversarial_examples dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    DEFAULT_CONFIG_NAME = "all"
    BUILDER_CONFIGS = [
        AdvWhisperASRConfig(name="targeted", description="Targeted adversarial examples, with target 'OK Google, browse to evil.com'"),
        AdvWhisperASRConfig(name="untargeted-35", description="Untargeted adversarial examples of radius approximately 35dB"),
        AdvWhisperASRConfig(name="untargeted-40", description="Untargeted adversarial examples of radius approximately 40dB"),
        AdvWhisperASRConfig(name="language-armenian", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Armenian"),
        AdvWhisperASRConfig(name="language-lithuanian", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Lithuanian"),
        AdvWhisperASRConfig(name="language-czech", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Czech"),
        AdvWhisperASRConfig(name="language-danish", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Danish"),
        AdvWhisperASRConfig(name="language-indonesian", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Indonesian"),
        AdvWhisperASRConfig(name="language-italian", description="Adversarial examples generated by fooling the whisper language detection module. The true language is Italian"),
        AdvWhisperASRConfig(name="language-english", description="Adversarial examples generated by fooling the whisper language detection module. The true language is English")
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(_DL_URLS[self.config.name])
        # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else {}
        models = [
            'whisper-tiny',
            'whisper-tiny.en',
            'whisper-base',
            'whisper-base.en',
            'whisper-small',
            'whisper-small.en',
            'whisper-medium',
            'whisper-medium.en',
            'whisper-large',
        ]
        seeds = {
            "targeted":2000,
            "untargeted-35": 235,
            "untargeted-40":240,
            "language-armenian":1030,
            "language-lithuanian":1030,
            "language-czech":1030,
            "language-danish":1030,
            "language-indonesian":1030,
            "language-italian":1030,
            "language-english":1030
        }
        folders = {
             "targeted":"cw",
            "untargeted-35": "pgd-35",
            "untargeted-40":"pgd-40",
            "language-armenian":"hy-AM",
            "language-lithuanian":"lt",
            "language-czech":"cs",
            "language-danish":"da",
            "language-indonesian":"id",
            "language-italian":"it",
            "language-english":"en"
        }
        targets = [("english","en"), ("tagalog","tl"), ("serbian","sr")]
        
        if "language-" in self.config.name:
            lang = self.config.name.split("language-")[-1]
            splits = [
                datasets.SplitGenerator(
                    name=lang+"."+target[0],
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("all"),
                        "files": dl_manager.iter_archive(archive_path["all"]),
                        "path_audio": os.path.join(folders[self.config.name]+"-"+target[1],"whisper-medium",str(seeds[self.config.name]),"save")
                    },
                ) for target in targets
            ] + [
                datasets.SplitGenerator(
                    name="original",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("all"),
                        "files": dl_manager.iter_archive(archive_path["all"]),
                        "path_audio": folders[self.config.name]+"-original"
                    },
                )
            ]
        else:
            splits = [
            datasets.SplitGenerator(
                    name=model.replace("-","."),
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("all"),
                        "files": dl_manager.iter_archive(archive_path["all"]),
                        "path_audio": os.path.join(folders[self.config.name],model,str(seeds[self.config.name]),"save")
                    },
                ) for model in models
            ] + [
                datasets.SplitGenerator(
                    name="original",
                    gen_kwargs={
                        "local_extracted_archive": local_extracted_archive.get("all"),
                        "files": dl_manager.iter_archive(archive_path["all"]),
                        "path_audio": os.path.join(folders[self.config.name],"original")
                    },
                )
            ]

        return splits

    def _generate_examples(self, files, local_extracted_archive,path_audio):
        """Generate examples from an extracted path."""
        key = 0
        suffix = "_nat.wav" if "original" in path_audio else "_adv.wav"
        audio_data = {}
        transcripts = []
        for t in files:
            path, f = t
            if path.endswith(".wav"):
                if path_audio in path and path.endswith(suffix):
                    id_ = path.split("/")[-1][: -len(suffix)]
                    audio_data[id_] = f.read()
            elif path.endswith(".csv"):
                for line in f:
                    if line:
                        line = (line.decode("utf-8") if isinstance(line,bytes) else line)
                        line=line.strip().split(",")
                        id_ = line[0]
                        transcript=line[-1]
                        transcript = transcript[:-1] if transcript[-1]=='\n' else transcript
                        audio_file = id_+suffix
                        audio_file = (
                            os.path.join(local_extracted_archive,path_audio, audio_file)
                            if local_extracted_archive else audio_file
                        )
                        transcripts.append(
                            {
                                "id": id_,
                                "file": audio_file,
                                "text": transcript,
                            }
                        )
        
        for transcript in transcripts:
            if transcript["id"] in audio_data:
                audio = {"path": transcript["file"], "bytes": audio_data[transcript["id"]]}
                yield key, {"audio": audio, **transcript}
                key += 1
        audio_data = {}
        transcripts = []