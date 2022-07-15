#!/bin/bash
DIR="../../results/bert_large/prof" # "../../results/bert_96/prof"
SUFFIX="_seqlen512"
#-------------------------
echo "create non-suffix symlink for ${DIR}/*${SUFFIX}*"
pushd . && cd ${DIR}
FILES=$(find . -maxdepth 1 -name "*${SUFFIX}*")
for file in $FILES
do
	LINK=${file/${SUFFIX}/""}
	echo "$file <-- ${LINK}"
	ln -s $file ${LINK}
	# rm $LINK
done
popd
tree $DIR
# (ref: https://stackoverflow.com/questions/13210880/replace-one-substring-for-another-string-in-shell-script)
