# GitHub 推进清单（当前项目）

用于在推送前快速确认：哪些内容建议提交、哪些内容建议保留在本地。

## 1) 建议提交（核心成果）

- `src/`：RAG 主链路、MCP Server、Dashboard、评测模块代码
- `scripts/`：数据导入、查询、评测与实验分析脚本
- `tests/`：单测/集成测试/端到端测试与 fixture
- `config/`：提示词与默认配置（不含密钥）
- `deploy/modelscope/`：部署入口及依赖定义
- `docs/`：实验报告、踩坑复盘、截图说明
- `README.md`、`DEV_SPEC.md`、`pyproject.toml`、`.env.example`

## 2) 不建议提交（本地产物）

- 本地环境与缓存：`.venv*`、`__pycache__/`、`.pytest_cache/`、`logs/`
- 本地数据库与索引：`data/`
- 批量实验中间结果目录：`results/*/`
- 临时材料与草稿产物：`artifacts/`
- IDE 本地配置：`.cursor/`
- 根目录临时图片：`62d3c30086ca571f3b1feaeff48d2725.jpg`、`d7bea6bc1319c6387c44278d408d48b8.jpg`

## 3) 推送前检查（必做）

1. **敏感信息检查**
   - 确认未提交 `.env`、密钥、token、个人路径信息。
2. **变更分组**
   - 建议至少分两批提交：
     - A: 代码能力升级（`src/` + `scripts/` + `tests/`）
     - B: 文档与配置（`README.md` + `docs/` + `config/`）
3. **最小验证**
   - 运行：
     - `pytest -q`
     - `python main.py`（服务能启动）
     - `python scripts/start_dashboard.py`（Dashboard 能启动）

## 4) 推荐提交顺序（示例）

```bash
git add .gitignore docs/GITHUB_PUSH_CHECKLIST.md
git add src scripts tests
git commit -m "feat: finalize modular RAG pipeline and evaluation tooling"

git add README.md docs config pyproject.toml .env.example
git commit -m "docs: publish experiment reports and usage guide"
```

## 5) 推送命令

```bash
git push origin <your-branch>
```

如果还没建远端分支：

```bash
git push -u origin HEAD
```
