---
trigger: always_on
---
1、执行命令脚本的话，请在 LIVI 虚拟环境下面运行，使用命令 conda activate LIVI ，激活 LIVI 虚拟环境。

2、模型训练的运行启动命令，请使用后台nohup运行，避免关掉终端后，训练被中断

3、多层包装会导致多层缓冲，使日志无法实时查看，Python 默认缓冲会延迟日志输出，长时间运行的任务必须使用 -u 参数

# 避免：多层嵌套
nohup conda run -n env bash -c "python script.py" > log 2>&1 &
# 推荐：直接调用
nohup /path/to/env/bin/python -u script.py > log 2>&1 &

4、后台任务应该直接使用 conda 环境的 Python 路径，conda run 适合临时命令，不适合长时间后台任务

5、复杂的启动命令容易忘记，写成脚本或文档，方便复用
