# mahjong

日本式リーチ麻雀 AI エージェント (`mahjong_agent`) の開発リポジトリ。

`riichi.dev` のルール・接続方式に対応する強い麻雀 AI を目指す。学習・推論のためのゲーム環境としては
[RiichiEnv](https://pypi.org/project/riichienv/) を外部依存として利用する。

## Package

- distribution name: `mahjong-agent`
- import name: `mahjong_agent`
- Python: `>=3.10,<3.15`

## Layout

```text
src/mahjong_agent/   # Python package source (src layout)
configs/             # training / evaluation YAML configs
tests/               # pytest 用テスト
```

実験 (runbook / report / driver) は本リポジトリには置かない。

## Install

開発用に editable install + dev extras を入れる。

```bash
python3 -m pip install -e ".[dev]"
```

## Test

```bash
python3 -m pytest
```

## Lint

```bash
python3 -m ruff check .
```
