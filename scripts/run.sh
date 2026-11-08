: ${IMAGE:=asssaf/ace-step-intel-xpu}

docker run --rm -it --name ace-step \
	--device /dev/dri:/dev/dri \
	--group-add 28 \
	--group-add 27 \
	-v /var/cache/models/acestep/checkpoints:/ACE-Step-1.5/checkpoints \
	-v "${PWD}/.env:/ACE-Step-1.5/.env:ro" \
	-p 7860:7860 \
	--net host \
	--ipc=host \
	"${IMAGE}"
