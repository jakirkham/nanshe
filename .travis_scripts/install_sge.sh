#!/bin/bash

export USER=$(whoami)
export SGE_CONFIG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export SGE_ROOT=/var/lib/gridengine
echo $SGE_CONFIG_DIR
sudo sed -i -r "s/^(127.0.0.1\s)(localhost\.localdomain\slocalhost)/\1localhost localhost.localdomain ${HOSTNAME} /" /etc/hosts
sudo cp /etc/resolv.conf /etc/resolv.conf.orig
sudo echo "domain ${HOSTNAME}" >> /etc/resolv.conf
echo "gridengine-master shared/gridenginemaster string ${HOSTNAME}" | sudo debconf-set-selections
echo "gridengine-master shared/gridenginecell string default" | sudo debconf-set-selections
echo "gridengine-master shared/gridengineconfig boolean true" | sudo debconf-set-selections
sudo cp ${SGE_ROOT}/default/common/act_qmaster ${SGE_ROOT}/default/common/act_qmaster.orig
sudo bash -c "echo $HOSTNAME > ${SGE_ROOT}/default/common/act_qmaster"
sudo service gridengine-master restart
sudo service gridengine-exec restart
export CORES=$(grep -c '^processor' /proc/cpuinfo)
sudo cp $SGE_CONFIG_DIR/user.conf.tmpl $SGE_CONFIG_DIR/user.conf
sudo sed -i -r "s/template/${USER}/" $SGE_CONFIG_DIR/user.conf
qconf -suserl | xargs -r -I {} sudo qconf -du {} arusers
qconf -suserl | xargs -r sudo qconf -duser
sudo qconf -Auser $SGE_CONFIG_DIR/user.conf
sudo qconf -au $USER arusers
qconf -ss | xargs -r sudo qconf -ds
qconf -sel | xargs -r sudo qconf -de
sudo qconf -as $HOSTNAME
sudo cp $SGE_CONFIG_DIR/host.conf.tmpl $SGE_CONFIG_DIR/host.conf
sudo sed -i -r "s/localhost/${HOSTNAME}/" $SGE_CONFIG_DIR/host.conf
export HOST_IN_SEL=$(qconf -sel | grep -c "${HOSTNAME}")
if [ $HOST_IN_SEL != "1" ]; then sudo qconf -Ae $SGE_CONFIG_DIR/host.conf; else sudo qconf -Me $SGE_CONFIG_DIR/host.conf; fi
sudo cp $SGE_CONFIG_DIR/queue.conf.tmpl $SGE_CONFIG_DIR/queue.conf
sudo sed -i -r "s/localhost/${HOSTNAME}/" $SGE_CONFIG_DIR/queue.conf
sudo sed -i -r "s/UNDEFINED/${CORES}/" $SGE_CONFIG_DIR/queue.conf
sudo cp $SGE_CONFIG_DIR/batch.conf.tmpl $SGE_CONFIG_DIR/batch.conf
qconf -sql | xargs -r sudo qconf -dq
qconf -spl | grep -v "make" | xargs -r sudo qconf -dp
sudo qconf -Ap $SGE_CONFIG_DIR/batch.conf
sudo qconf -Aq $SGE_CONFIG_DIR/queue.conf
sudo service gridengine-master restart
sudo service gridengine-exec restart
echo "Printing queue info to verify that things are working correctly."
qstat -f -q all.q -explain a
echo "You should see sge_execd and sge_qmaster running below:"
ps aux | grep "sge"
# Add a job based test to make sure the system really works.
echo
echo "Submit a simple job to make sure the submission system really works."

mkdir /tmp/test_gridengine &>/dev/null
pushd /tmp/test_gridengine &>/dev/null
set -e

echo "-------------- test.sh --------------"
echo -e '#!/bin/bash\necho "stdout"\necho "stderr" 1>&2' | tee test.sh
echo "-------------------------------------"
echo
chmod +x test.sh

qsub -cwd -sync y test.sh
echo

echo "------------ test.sh.o1 -------------"
cat test.sh.o*
echo "-------------------------------------"
echo

echo "------------ test.sh.e1 -------------"
cat test.sh.e*
echo "-------------------------------------"
echo

grep stdout test.sh.o* &>/dev/null
grep stderr test.sh.e* &>/dev/null

rm test.sh*

set +e
popd &>/dev/null
rm -rf /tmp/test_gridengine &>/dev/null
