# AIRBOT 5.2.2 安装包

为了让 `./install.sh` 无需使用某台电脑的绝对路径，请将以下厂商软件包放在本目录：

```text
arm_sdk-5.2.2-py3-none-any.whl
airbot-arm_5.2.2_amd64.deb
```

厂商二进制包不由本项目生成。也可以不复制到这里，改用：

```bash
./install.sh --sdk-wheel /path/to/arm_sdk-5.2.2-py3-none-any.whl \
  --arm-deb /path/to/airbot-arm_5.2.2_amd64.deb
```
