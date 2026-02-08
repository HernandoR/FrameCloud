# Benchmark CI Artifacts Implementation

**Date**: 2026-02-08  
**Agent**: GitHub Copilot (Claude Sonnet 4.5)

## 概述

修改了 benchmark CI 工作流，使其能够上传生成的图表和结果文件作为 GitHub Actions artifacts，用户可以通过 workflow run 页面的链接下载。

## 修改内容

### 1. test-slow-ubuntu.yml

在 Ubuntu benchmark 测试工作流中添加了两个新步骤：

#### Upload benchmark results

- 使用 `actions/upload-artifact@v4` 上传 benchmark 结果
- **Artifact 名称**: `benchmark-results-ubuntu`
- **上传内容**:
  - JSON 结果文件: `reports/benchmarks/*.json`
  - SVG 图表文件: `reports/benchmarks/*.svg`
  - 平台特定结果目录: `Darwin-CPython-*/`, `Linux-*/`, `Windows-*/`
- **保留时间**: 30 天
- **执行条件**: `if: always()` - 即使测试失败也会上传

#### Generate benchmark summary

- 在 GitHub Actions Summary 页面生成摘要信息
- 列出所有生成的 SVG 图表文件
- 提供下载提示信息

### 2. test-slow-cross-platform.yml

在跨平台 benchmark 测试工作流中添加了相同的步骤，但针对 matrix 策略做了优化：

#### Upload benchmark results

- **Artifact 名称**: `benchmark-results-${{ matrix.os }}` (根据操作系统动态命名)
  - macOS: `benchmark-results-macos-latest`
  - Windows: `benchmark-results-windows-latest`
- 其他配置与 Ubuntu 版本相同

#### Generate benchmark summary

- 标题包含操作系统信息: `Benchmark Results (${{ matrix.os }})`
- 其他功能与 Ubuntu 版本相同

## 技术细节

### 使用的 GitHub Actions

- **actions/upload-artifact@v4**: 最新版本的 artifact 上传 action
  - 支持 Node.js 24 运行时
  - 提供更快的上传速度（相比 v3 提升高达 90%）
  - 自动在 UI 和 REST API 中提供 artifact ID
  - 支持设置保留时间、压缩级别等选项

### Artifact 结构

上传的 artifacts 包含以下内容：

```
benchmark-results-ubuntu/
├── benchmark.json              # 完整的 benchmark 结果数据
├── histogram-*.svg             # 各类别的 histogram 图表
│   ├── histogram-attributes.svg
│   ├── histogram-creation.svg
│   ├── histogram-io.svg
│   ├── histogram-large-scale.svg
│   ├── histogram-sampling.svg
│   ├── histogram-transformation.svg
│   └── histogram-voxelmap.svg
└── Linux-CPython-3.12-64bit/   # 平台特定的详细结果
    └── *.json
```

### 访问下载链接

用户可以通过以下方式访问和下载 benchmark 结果：

1. **Workflow Run 页面**:
   - 打开 GitHub Actions 页面
   - 进入对应的 workflow run
   - 在页面底部的 "Artifacts" 部分找到上传的文件
   - 点击 artifact 名称即可下载

2. **GitHub Actions Summary**:
   - 在 workflow run 页面查看 Summary
   - 会显示生成的图表列表和下载提示

3. **REST API**:
   - 可通过 GitHub REST API 编程方式访问
   - Artifact 上传时会返回 artifact-id 和 artifact-url

## 优点

1. **自动化**: 每次运行 benchmark 测试后自动上传结果
2. **持久化**: 结果保留 30 天，方便后续查看和对比
3. **可访问性**: 通过简单的链接即可下载，无需克隆仓库
4. **平台区分**: 跨平台测试的结果按操作系统分别保存
5. **容错性**: 使用 `if: always()` 确保即使测试失败也能上传部分结果
6. **可见性**: GitHub Actions Summary 提供清晰的摘要信息

## 注意事项

1. **存储限制**: GitHub Actions 有存储配额限制，注意监控使用量
2. **网络依赖**: 下载 artifacts 需要登录 GitHub 账号
3. **保留时间**: artifacts 默认保留 30 天，之后会被自动删除
4. **权限要求**: 仓库的 write 权限用户才能删除 artifacts

## 相关文档

- [GitHub Actions: Upload Artifact](https://github.com/actions/upload-artifact)
- [GitHub Docs: Storing workflow data as artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
