# ADK 2.0 Graph Workflow 入門

- **Tags**: ADK, Graph Workflow, Gemini, Agent Development Kit

## 概要
ADK (Agent Development Kit) 2.0 の Graph `Workflow` API は、ノード（Nodes）とエッジ（Edges）を用いて明示的な制御フローを持つエージェントを構築するための仕組みです。

## 主な構成要素
1. **`START`**: ワークフローのエントリーポイントであり、ユーザー入力を受け取ります。
2. **Nodes（ノード）**:
   - `FunctionNode`（通常の Python 関数）: バリデーション、ルーティング判定、ファイル操作などの決定論的な処理を実行します。
   - `LlmAgent`: Gemini モデルを用いた推論やテキスト生成を行います。ワークフロー内で構造化データを後続ノードに渡す際は Pydantic の `output_schema` を指定します。
3. **Conditional Edges（条件付きエッジ）**:
   - ノードが `Event(route="route_name", output=...)` または `EventActions(route="route_name")` を返すことで、後続の分岐先ノードを動的に切り替えることができます。
   - ガードレール（特定トピック以外の遮断など）をプロンプト任せにせず、グラフ構造として強制できるのが大きな利点です。
