FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

# install basic tools
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install \
    vim wget curl git \
    libopencv-dev \
    python3 python3-pip 

# install oh my zsh & change theme to af-magic
RUN apt-get -y install zsh 
COPY ./zsh-install.sh ./
RUN sh ./zsh-install.sh && \
    sed -i 's/ZSH_THEME=\"[a-z0-9\-]*\"/ZSH_THEME="af-magic"/g' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/g' ~/.zshrc  && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    rm ./zsh-install.sh 
COPY ./requirements.txt ./
RUN chsh root -s /bin/zsh && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r ./requirements.txt && \
    rm -f ./requirements.txt
