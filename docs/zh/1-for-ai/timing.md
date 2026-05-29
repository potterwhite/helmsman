# 计时系统详解

## 两种计时器

### ScopedTimer
- 作用域计时器，构造时 start，析构时自动 record
- 用于函数级计时
- 示例：`ScopedTimer t("name", timing_enabled, logger, module);`

### StageAccumulator (sa)
- 手动累计计时器，需要显式 start/stop/record
- 用于循环内每帧计时，最后 report 一次
- 示例：`sa acc_{"name"}; acc_.record(elapsed_ms);`

## 嵌套关系

```
Lv01::pipeline.run() total          (pipeline.cpp, ScopedTimer)
  └── Lv02::pipeline::RVMMode::run() total          (rvm.cpp, ScopedTimer)
        └── Lv02-01::main::loop_total     (rvm.cpp, StageAccumulator, 每帧)
              ├── Lv02-01-01::worker::preprocess  (FrontendBase, StageAccumulator)
              ├── Lv02-01-02::main::decode         (rvm.cpp, StageAccumulator)
              ├── Lv02-01-03::main::infer          (rvm.cpp, StageAccumulator)
              └── Lv02-01-04::main::composite      (rvm.cpp, StageAccumulator)
                    ├── resize_alpha
                    ├── resize_frame
                    ├── blend
                    ├── upscale
                    ├── writer
                    └── drm_show
```

## 命名约定

- `Lv01` / `Lv02` / `Lv03` — 嵌套层级
- `main` — 主线程
- `worker` — 工作线程
- `comp` — composite 子阶段
- 缩进用空格表示层级关系

## 报告时机

- ScopedTimer：析构时自动报告（如果 timing_enabled）
- StageAccumulator：调用 `report()` 时报告，通常在主循环结束后
