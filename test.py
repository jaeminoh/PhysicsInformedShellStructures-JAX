from jaxtyping import install_import_hook

hook = install_import_hook("src", "typeguard.typechecked")
import src
import src._geometry
hook.uninstall()