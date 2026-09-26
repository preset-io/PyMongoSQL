// Fork publisher. Pull-request wheels are versioned <version>+pr.<number>.<sha>
// (PEP 440-normalized by ci/release_version.py); stable wheels are published only
// from reviewed master. A published artifact is never overwritten.
podTemplate(
    imagePullSecrets: ['preset-pull'],
    containers: [
        containerTemplate(name: 'ci', image: 'preset/ci:latest',
            ttyEnabled: true, command: 'cat'),
        containerTemplate(name: 'py-ci', image: 'preset/python:3.9.18-2024-02-21-ci',
            ttyEnabled: true, command: 'cat'),
        // Disposable server for the test suite; reachable on localhost inside the pod.
        containerTemplate(name: 'mongo', image: 'mongo:8.0',
            envVars: [
                envVar(key: 'MONGO_INITDB_ROOT_USERNAME', value: 'admin'),
                envVar(key: 'MONGO_INITDB_ROOT_PASSWORD', value: 'secret'),
            ])
    ]
) {
    node(POD_LABEL) {
        checkout scm
        def revision = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()
        boolean isMaster = env.BRANCH_NAME == 'master'
        boolean isPR = env.CHANGE_ID != null
        if (!isMaster && !isPR) {
            error('Only master and pull-request builds publish; use a PR.')
        }

        container('py-ci') {
            stage('Test and build') {
                def args = isMaster ? '' : "${env.CHANGE_ID} ${revision.take(12)}"
                sh '''
                    set -eu
                    python -m venv .venv
                    .venv/bin/pip install 'sqlalchemy==2.0.52' 'pymongo==4.17.0' \
                        'antlr4-python3-runtime==4.13.2' 'jmespath==1.1.0' 'pandas>=2.2,<3' \
                        'tenacity==9.1.2' 'sqlglot==30.18.0' 'pytest==8.3.5' 'boto3>=1.36,<2' 'packaging==25.0' \
                        'build==1.4.4' 'setuptools==80.9.0' 'setuptools_scm==8.3.1' 'wheel==0.45.1'
                '''
                def version = sh(script: ".venv/bin/python ci/release_version.py ${args}",
                    returnStdout: true).trim()
                env.PUBLISH_VERSION = version
                env.WHEEL = "pymongosql-${version}-py3-none-any.whl"
                env.KEY = "pymongosql/${env.WHEEL}"
                sh '''
                    set -eu
                    # The checkout is owned by another uid; setuptools_scm runs git during builds.
                    # The pod is ephemeral, so its global git config is disposable.
                    git --version
                    git config --global --add safe.directory "$PWD"
                    .venv/bin/pip install --no-deps -e .
                    for attempt in $(seq 1 30); do
                        .venv/bin/python -c "import pymongo; pymongo.MongoClient('mongodb://admin:secret@localhost:27017', serverSelectionTimeoutMS=2000).admin.command('ping')" && break
                        sleep 2
                    done
                    .venv/bin/python tests/run_test_server.py setup
                    .venv/bin/python -m pytest -q tests
                    .venv/bin/pip uninstall -y pymongosql
                    python - <<'PY'
import os
import re
from pathlib import Path
path = Path('pymongosql/__init__.py')
source = path.read_text()
pattern = re.compile(r'^__version__: str = "[^"]+"$', re.MULTILINE)
assert len(pattern.findall(source)) == 1
path.write_text(pattern.sub('__version__: str = "' + os.environ['PUBLISH_VERSION'] + '"', source))
PY
                    SOURCE_DATE_EPOCH=$(git -c safe.directory="$PWD" log -1 --format=%ct)
                    case "$SOURCE_DATE_EPOCH" in
                        ''|*[!0-9]*) echo "Invalid commit timestamp for reproducible build" >&2; exit 1 ;;
                    esac
                    export SOURCE_DATE_EPOCH
                    # Pin the build backend and remove stale output for reproducible retries.
                    rm -rf build dist pymongosql.egg-info
                    .venv/bin/python -m build --wheel --no-isolation
                    test -f "dist/$WHEEL" || { echo "missing dist/$WHEEL"; ls -1 dist; exit 1; }
                    .venv/bin/python -c "import os, sys, zipfile; names = zipfile.ZipFile('dist/' + os.environ['WHEEL']).namelist(); sys.exit('wheel ships tests or ci' if any(n.startswith(('tests/', 'ci/')) for n in names) else 0)"
                    .venv/bin/pip install --force-reinstall --no-deps "dist/$WHEEL"
                    .venv/bin/python - <<'PY'
import importlib.metadata as im
import os
import sqlalchemy as sa
assert im.version('pymongosql') == os.environ['PUBLISH_VERSION']
engine = sa.create_engine('mongodb://user@localhost/db')
assert engine.dialect.name == 'mongodb'
engine.dispose()
PY
                    sha256sum "dist/$WHEEL"
                '''
            }
        }
        container('ci') {
            stage('Publish immutable wheel') {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: 'ci-user',
                    accessKeyVariable: 'AWS_ACCESS_KEY_ID',
                    secretKeyVariable: 'AWS_SECRET_ACCESS_KEY'
                ]]) {
                    withEnv(["ALLOW_IDENTICAL_PR_ARTIFACT=${isPR && !isMaster}"]) {
                        sh '''
                            set -eu
                            python -m pip install --quiet 'boto3>=1.36,<2'
                            python ci/publish_wheel.py
                        '''
                    }
                }
            }
        }
        archiveArtifacts artifacts: 'dist/*.whl,published.sha256', fingerprint: true
    }
}
