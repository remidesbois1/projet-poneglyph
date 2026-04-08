import "./robin.css";

export const metadata = {
    title: "Bureau de Robin | Investigation",
    description: "Interrogez l'archive Poneglyph. Un agent IA fouille les pages de manga pour répondre à vos questions avec des preuves sourcées.",
};

import RobinBoard from "@/components/robin/RobinBoard";

export default function ChatPage() {
    return <RobinBoard />;
}
